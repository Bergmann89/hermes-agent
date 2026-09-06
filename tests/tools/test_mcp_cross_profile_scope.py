"""MCP tools must reach EVERY served profile's session, not just the first-discovered one.

Under gateway multiplexing MCP connection state is process-global (keyed by server name) but
registry entries are per-HERMES_HOME scoped overlays. The first profile to connect a server used to
register its tools into ONLY that profile's overlay; later profiles' discovery short-circuited on the
process-global name dedup and never populated their own overlay, so the tools were neither
model-facing nor deferrable for them (#67605). register_connected_into_current_scope heals this by
re-registering an already-connected server's live tools into the current profile scope (idempotent,
no new subprocess); teardown then deregisters from every scope the server registered into.
"""
from __future__ import annotations

import pytest


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"desc {name}"
        self.input_schema = {"type": "object", "properties": {}}
        self.annotations = None


class _FakeServer:
    """Minimal stand-in for a connected MCPServerTask: has a live session and _tools."""
    def __init__(self, name, tool_names):
        self.name = name
        self.session = object()  # non-None => connected
        self._tools = [_FakeTool(t) for t in tool_names]
        self.tool_timeout = 30
        self._registered_tool_names = []


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    from agent import secret_scope as ss
    from tools import mcp_tool as _core
    from tools.registry import registry
    from hermes_constants import set_hermes_home_override
    monkeypatch.setattr(_core, "_MCP_AVAILABLE", True, raising=False)
    ss.set_multiplex_active(True)

    def _clear():
        for m in ("_servers", "_server_scope_keys", "_server_tool_scopes",
                  "_server_config_fingerprints", "_mcp_tool_server_names"):
            getattr(_core, m).clear()
        # Drop any mcp-* registry entries this test registered into scope overlays, so the
        # process-global registry is not polluted for other tests (the real runner isolates per
        # file, but keep the shared-state footprint clean regardless).
        for scope_map in list(registry._scoped_tools.values()):
            for tool_name in [n for n, e in scope_map.items() if e.toolset.startswith("mcp-")]:
                scope_map.pop(tool_name, None)

    _clear()
    try:
        yield
    finally:
        ss.set_multiplex_active(False)
        set_hermes_home_override(None)  # _scope() leaks a HERMES_HOME override ContextVar; reset it
        _clear()


def _scope(path):
    """Bind HERMES_HOME override so registry.current_scope_key() == this profile; return the key."""
    from hermes_constants import set_hermes_home_override, hermes_home_key
    set_hermes_home_override(str(path))
    return hermes_home_key(str(path))


def _tools_in_scope(server_name):
    from tools.registry import registry
    return registry.get_tool_names_for_toolset(f"mcp-{server_name}")


def test_non_owning_profile_gets_tools_after_reregister(tmp_path):
    # Profile A "connects" the server and registers under A's scope.
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    srv = _FakeServer("cbm", ["list_projects", "search_code"])
    _core._servers["cbm"] = srv

    a = tmp_path / "A"
    _scope(a)
    assert reg.register_connected_into_current_scope({"cbm": {}}) == 1
    assert any("list_projects" in t for t in _tools_in_scope("cbm")), _tools_in_scope("cbm")

    # Profile B: same process, its own scope, empty overlay -> must heal.
    b = tmp_path / "B"
    _scope(b)
    assert _tools_in_scope("cbm") == []  # B starts empty (the bug)
    assert reg.register_connected_into_current_scope({"cbm": {}}) == 1
    got = _tools_in_scope("cbm")
    assert any("list_projects" in t for t in got), got
    assert any("search_code" in t for t in got), got


def test_idempotent_double_register(tmp_path):
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    _core._servers["cbm"] = _FakeServer("cbm", ["list_projects"])
    _scope(tmp_path / "B")
    assert reg.register_connected_into_current_scope({"cbm": {}}) == 1
    # Second call: scope already has the tools -> no-op.
    assert reg.register_connected_into_current_scope({"cbm": {}}) == 0


def test_sr1_server_absent_from_profile_config_not_registered(tmp_path):
    # A server connected process-wide but NOT in profile B's config dict must not land in B's scope.
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    _core._servers["other"] = _FakeServer("other", ["secret_tool"])
    _scope(tmp_path / "B")
    # B's config only lists 'cbm' (which isn't connected) — 'other' must be ignored.
    assert reg.register_connected_into_current_scope({"cbm": {}}) == 0
    assert _tools_in_scope("other") == []


def test_teardown_deregisters_all_scopes(tmp_path):
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    from tools.registry import registry
    srv = _FakeServer("cbm", ["list_projects"])
    _core._servers["cbm"] = srv
    ka = _scope(tmp_path / "A"); reg.register_connected_into_current_scope({"cbm": {}})
    kb = _scope(tmp_path / "B"); reg.register_connected_into_current_scope({"cbm": {}})
    # Both scopes tracked.
    assert _core._server_tool_scopes["cbm"] == {ka, kb}
    # Deregister the one tool from all scopes.
    tool_name = registry.get_tool_names_for_toolset("mcp-cbm")[0]
    reg.deregister_mcp_tool_all_scopes("cbm", tool_name)
    _scope(tmp_path / "A"); assert _tools_in_scope("cbm") == []
    _scope(tmp_path / "B"); assert _tools_in_scope("cbm") == []


def test_no_duplicate_subprocess(tmp_path):
    # Healing must reuse the one live connection, never add a second _servers entry.
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    _core._servers["cbm"] = _FakeServer("cbm", ["list_projects"])
    _scope(tmp_path / "A"); reg.register_connected_into_current_scope({"cbm": {}})
    _scope(tmp_path / "B"); reg.register_connected_into_current_scope({"cbm": {}})
    assert list(_core._servers) == ["cbm"]  # still exactly one connection


def test_reload_selection_unaffected(tmp_path):
    # _server_scope_keys (connection-lifecycle selection for /reload-mcp) stays single-owning-scope;
    # the multi-scope tracking lives in the SEPARATE _server_tool_scopes map.
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    ka = _scope(tmp_path / "A")
    _core._servers["cbm"] = _FakeServer("cbm", ["list_projects"])
    _core._server_scope_keys["cbm"] = ka  # A owns the connection
    reg.register_connected_into_current_scope({"cbm": {}})
    _scope(tmp_path / "B"); reg.register_connected_into_current_scope({"cbm": {}})
    # Owning-scope selection unchanged (still A), even though tools live in {A, B}.
    assert _core._server_scope_keys["cbm"] == ka
    assert len(_core._server_tool_scopes["cbm"]) == 2


def test_no_op_outside_multiplex(tmp_path, monkeypatch):
    from agent import secret_scope as ss
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    ss.set_multiplex_active(False)
    _core._servers["cbm"] = _FakeServer("cbm", ["list_projects"])
    # No multiplex -> single global overlay already holds everything -> helper is a no-op.
    assert reg.register_connected_into_current_scope({"cbm": {}}) == 0


def test_route_divergence_fails_closed(tmp_path):
    # Profile A owns a 'cbm' connection over its own ssh key; profile B configures the same NAME
    # but a different route. B must NOT borrow A's connection/identity (fingerprint mismatch).
    from tools import mcp_tool_registration as reg
    from tools import mcp_tool as _core
    from tools.mcp_schema_cache import config_fingerprint
    cfg_a = {"command": "ssh", "args": ["-i", "~/.ssh/felix_ed25519", "felix@h", "cbm-mcp"]}
    cfg_b = {"command": "ssh", "args": ["-i", "~/.ssh/jonas_ed25519", "jonas@h", "cbm-mcp"]}
    _core._servers["cbm"] = _FakeServer("cbm", ["list_projects"])
    _core._server_config_fingerprints["cbm"] = config_fingerprint(cfg_a)  # A opened it
    _scope(tmp_path / "B")
    # B's config routes differently -> heal refuses, scope stays empty.
    assert reg.register_connected_into_current_scope({"cbm": cfg_b}) == 0
    assert _tools_in_scope("cbm") == []
    # Same route -> heal proceeds.
    assert reg.register_connected_into_current_scope({"cbm": cfg_a}) == 1
    assert any("list_projects" in t for t in _tools_in_scope("cbm"))

