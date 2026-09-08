"""Route-dependent MCP state must be keyed by the composite route identity ``(name, fingerprint)``.

Two profiles in one process can configure the SAME server name over DIFFERENT routes (e.g. ssh
with different keys). Every per-route map — the connect dedup/cooldown/error set, the circuit
breaker and the trust gate — must isolate those routes so one profile's failing/untrusted route
never blocks, opens the breaker for, or removes the gate from another profile's same-name route.

Companion to test_mcp_divergent_route_per_profile.py (connection identity) and
test_mcp_cross_profile_scope.py (registry-scope heal). Reviewer ehz0ah's REQUEST_CHANGES items.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool as _core
from tools import mcp_tool_discovery as _discovery
from tools import mcp_tool_handlers as _handlers
from tools import mcp_tool_lifecycle as _lifecycle
from tools import mcp_tool_loop as _loop


CFG_A = {"command": "ssh", "args": ["-i", "~/.ssh/felix_ed25519", "felix@h", "cbm-mcp"]}
CFG_B = {"command": "ssh", "args": ["-i", "~/.ssh/jonas_ed25519", "jonas@h", "cbm-mcp"]}


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"desc {name}"
        self.input_schema = {"type": "object", "properties": {}}
        self.annotations = None


class _FakeServer:
    def __init__(self, name, tool_names, config):
        self.name = name
        self.session = object()  # non-None => connected
        self._tools = [_FakeTool(t) for t in tool_names]
        self.tool_timeout = 30
        self._registered_tool_names = list(tool_names)
        self._sampling = None
        self._config = config


@pytest.fixture(autouse=True)
def _reset():
    maps = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_mcp_tool_server_names",
            "_server_connecting", "_server_connect_errors", "_server_connect_retry_after",
            "_server_connect_failures", "_server_error_counts", "_server_breaker_opened_at",
            "_server_trust_levels", "_tool_read_only_hints")

    def _clear():
        for m in maps:
            getattr(_core, m).clear()

    _clear()
    try:
        yield
    finally:
        _clear()


# --------------------------------------------------------------------------- Blocker 1
def test_divergent_route_connecting_does_not_block_peer():
    """Route A already connecting (composite key in _server_connecting) must NOT exclude route B."""
    _core._server_connecting.add(_core._server_key("cbm", CFG_A))
    picked = _discovery._select_new_servers({"cbm": CFG_B})
    assert "cbm" in picked, "route B was wrongly skipped while route A was connecting"
    # Route B is now recorded under ITS OWN composite key, never A's.
    assert _core._server_key("cbm", CFG_B) in _core._server_connecting
    # Ownership recorded under B's OWN composite key (not A's, and not a bare name).
    assert _core._server_key("cbm", CFG_B) in _core._server_scope_keys
    # Same route (A) IS still deduped.
    _core._server_connecting.clear()
    _core._server_connecting.add(_core._server_key("cbm", CFG_A))
    assert _discovery._select_new_servers({"cbm": CFG_A}) == {}


def test_divergent_route_cooldown_does_not_block_peer():
    """Route A in connect-backoff must NOT keep route B out of the candidate set."""
    _discovery._record_connect_failure(_core._server_key("cbm", CFG_A))
    assert _discovery._connect_cooldown_active(_core._server_key("cbm", CFG_A)) is True
    picked = _discovery._select_new_servers({"cbm": CFG_B})
    assert "cbm" in picked, "route B was blocked by route A's cooldown"


# --------------------------------------------------------------------------- Blocker 3
def test_status_does_not_mix_A_ownership_with_B_runtime(monkeypatch):
    """This profile configures route B; a live route-A connection must not surface as B's runtime."""
    from tools import mcp_tool_config as _config
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"cbm": CFG_B})
    monkeypatch.setattr(_core, "_mcp_registry_scope", lambda: "profile:B")
    # Only route A is live, owned by another profile.
    _core._servers[_core._server_key("cbm", CFG_A)] = _FakeServer("cbm", ["felix_only"], CFG_A)
    _core._server_scope_keys[_core._server_key("cbm", CFG_A)] = "profile:A"

    [status] = _discovery.get_mcp_status()
    # B's route is not connected: no A runtime leaks in.
    assert status["status"] == "configured"
    assert status["connected"] is False
    assert status["tools"] == 0


# --------------------------------------------------------------------------- Blocker 4
def test_scoped_shutdown_clears_failed_route_state(monkeypatch):
    """A failed/connecting route (no live server) must have ALL its connection state cleared by a
    scope-restricted shutdown, so the next /reload-mcp attempt is not blocked."""
    monkeypatch.setattr(_loop, "_stop_mcp_loop", lambda **_kw: None)
    key_work = _core._server_key("cbm", CFG_A)
    key_other = _core._server_key("cbm", CFG_B)
    # 'work' route failed to connect: it owns a scope entry but has NO live server.
    _core._server_scope_keys[key_work] = "profile:work"
    _core._server_scope_keys[key_other] = "profile:other"
    _core._server_connecting.add(key_work)
    _core._server_connect_errors[key_work] = "work error"
    _core._server_connect_errors[key_other] = "other error"
    _core._server_connect_retry_after[key_work] = 1.0
    _core._server_connect_retry_after[key_other] = 2.0
    _core._server_connect_failures[key_work] = 1
    _core._server_connect_failures[key_other] = 2

    _lifecycle.shutdown_mcp_servers(scope="profile:work")

    with _core._lock:
        assert key_work not in _core._server_connecting
        assert key_work not in _core._server_connect_errors
        assert key_work not in _core._server_scope_keys
        assert key_work not in _core._server_connect_retry_after
        assert key_work not in _core._server_connect_failures
        # The other profile's route is untouched.
        assert _core._server_connect_errors == {key_other: "other error"}
        assert _core._server_scope_keys == {key_other: "profile:other"}
        assert _core._server_connect_retry_after == {key_other: 2.0}
        assert _core._server_connect_failures == {key_other: 2}


# --------------------------------------------------------------------------- Blocker 5
def test_reload_projects_composite_keys_to_names_without_typeerror():
    """The /reload-mcp diff joins server NAMES; projecting composite keys must not raise
    TypeError: sequence item 0: expected str instance, tuple found."""
    _core._servers[_core._server_key("cbm", CFG_A)] = _FakeServer("cbm", ["a"], CFG_A)
    _core._servers[_core._server_key("other", CFG_B)] = _FakeServer("other", ["b"], CFG_B)
    names = {_core._key_name(k) for k in _core._servers}
    # This is exactly what gateway/run_turn and cli_info_mixin now do before join.
    joined = ", ".join(sorted(names))
    assert joined == "cbm, other"


# --------------------------------------------------------------------------- Blockers 6 & 7 helpers
def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _run():
            for srv in list(_core._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_run())
    finally:
        loop.close()


class _Result:
    def __init__(self, text="ok"):
        self.content = [SimpleNamespace(text=text, type="text")]
        self.isError = False
        self.structuredContent = None


def _install_two_routes():
    """Seed two divergent-route live connections for the same name, each under its composite key."""
    for cfg, tool in ((CFG_A, "felix_tool"), (CFG_B, "jonas_tool")):
        session = MagicMock()
        session.call_tool = AsyncMock(return_value=_Result())
        srv = SimpleNamespace(session=session, _rpc_lock=None, name="cbm",
                              _config=cfg, _tools=[], _registered_tool_names=[tool])
        _core._servers[_core._server_key("cbm", cfg)] = srv


# --------------------------------------------------------------------------- Blocker 6
def test_circuit_breaker_isolated_per_route(monkeypatch):
    """Opening the breaker on route A must NOT short-circuit route B's same-name handler."""
    _install_two_routes()
    monkeypatch.setattr(_loop, "_run_on_mcp_loop", _fake_run_on_mcp_loop)
    monkeypatch.setattr(_discovery, "_get_connected_server_for_call",
                        lambda name: _core._servers[_core._server_key("cbm", CFG_B)])
    # Trip route A's breaker.
    _core._server_error_counts[_core._route_key("cbm", CFG_A)] = _core._CIRCUIT_BREAKER_THRESHOLD + 2
    import time as _t
    _core._server_breaker_opened_at[_core._route_key("cbm", CFG_A)] = _t.monotonic()

    # Route A handler: breaker is open -> short-circuits.
    handler_a = _handlers._make_tool_handler("cbm", "felix_tool", 10.0, CFG_A)
    assert "error" in json.loads(handler_a({})), "route A's breaker should be open"

    # Route B handler for the SAME server name: breaker is still CLOSED -> call goes through.
    handler_b = _handlers._make_tool_handler("cbm", "jonas_tool", 10.0, CFG_B)
    assert json.loads(handler_b({})) == {"result": "ok"}, "route B was wrongly gated by route A's breaker"


# --------------------------------------------------------------------------- Blocker 7
def test_trust_gate_isolated_per_route(monkeypatch):
    """A later trust:full same-name route must NOT strip an existing trust:untrusted route's gate."""
    _install_two_routes()
    monkeypatch.setattr(_loop, "_run_on_mcp_loop", _fake_run_on_mcp_loop)
    monkeypatch.setattr(_discovery, "_get_connected_server_for_call",
                        lambda name: _core._servers[_core._server_key("cbm", CFG_A)])
    # Route A is untrusted; a divergent route B later registers as full-trust.
    _core._server_trust_levels[_core._route_key("cbm", CFG_A)] = _core._TRUST_UNTRUSTED
    _core._server_trust_levels[_core._route_key("cbm", CFG_B)] = _core._TRUST_FULL

    # Route A's write-capable handler still gates (no readOnlyHint recorded => write-capable).
    handler_a = _handlers._make_tool_handler("cbm", "felix_tool", 10.0, CFG_A)
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as consent:
        raw = handler_a({"x": 1})
    consent.assert_called_once()
    assert "did not approve" in json.loads(raw)["error"]

    # Route B (full trust) never consults approval.
    monkeypatch.setattr(_discovery, "_get_connected_server_for_call",
                        lambda name: _core._servers[_core._server_key("cbm", CFG_B)])
    handler_b = _handlers._make_tool_handler("cbm", "jonas_tool", 10.0, CFG_B)
    with patch("tools.approval_prompt.request_elicitation_consent") as consent_b:
        assert json.loads(handler_b({})) == {"result": "ok"}
    consent_b.assert_not_called()


def test_trust_gate_fails_closed_on_missing_route(monkeypatch):
    """Missing trust for a route = full ONLY for that exact route; an unknown route stays default full,
    but a route explicitly untrusted keeps gating even when a same-name full route exists (fail closed)."""
    _install_two_routes()
    monkeypatch.setattr(_loop, "_run_on_mcp_loop", _fake_run_on_mcp_loop)
    monkeypatch.setattr(_discovery, "_get_connected_server_for_call",
                        lambda name: _core._servers[_core._server_key("cbm", CFG_A)])
    # Only route B recorded (full); route A has NO entry -> defaults to full for A's own key.
    _core._server_trust_levels[_core._route_key("cbm", CFG_B)] = _core._TRUST_FULL
    err = _handlers._trust_gate_check("cbm", "felix_tool", _core._route_key("cbm", CFG_A))
    assert err is None  # missing => full for that exact route
    # Now mark A untrusted: the same-name full route B must not lift A's gate.
    _core._server_trust_levels[_core._route_key("cbm", CFG_A)] = _core._TRUST_UNTRUSTED
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline"):
        blocked = _handlers._trust_gate_check("cbm", "felix_tool", _core._route_key("cbm", CFG_A))
    assert blocked is not None and "did not approve" in json.loads(blocked)["error"]
