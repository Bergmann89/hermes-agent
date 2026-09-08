"""Divergent-route same-name MCP servers get their OWN per-profile connection.

Case 2 of the per-profile design: two profiles configure the SAME server name (``cbm``) with
DIFFERENT routes (e.g. ssh with different keys). Because ``_servers`` is keyed by the composite
``(name, config_fingerprint)``:

- ``_select_new_servers`` must NOT drop the second route as a name-dedup (it spawns its own
  connection), and
- ``register_connected_into_current_scope`` must resolve each profile's OWN ``(name, fp)`` so scope A
  sees route A's tools and scope B sees route B's tools — never the peer's.
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
    def __init__(self, name, tool_names, config):
        self.name = name
        self.session = object()  # non-None => connected
        self._tools = [_FakeTool(t) for t in tool_names]
        self.tool_timeout = 30
        self._registered_tool_names = []
        self._config = config


CFG_A = {"command": "ssh", "args": ["-i", "~/.ssh/felix_ed25519", "-p", "10022", "felix@h", "cbm-mcp"]}
CFG_B = {"command": "ssh", "args": ["-i", "~/.ssh/jonas_ed25519", "-p", "10023", "jonas@h", "cbm-mcp"]}


@pytest.fixture(autouse=True)
def _reset():
    from agent import secret_scope as ss
    from tools import mcp_tool as _core
    from tools.registry import registry
    from hermes_constants import set_hermes_home_override
    ss.set_multiplex_active(True)

    def _clear():
        for m in ("_servers", "_server_scope_keys", "_server_tool_scopes",
                  "_mcp_tool_server_names", "_server_connecting"):
            getattr(_core, m).clear()
        for scope_map in list(registry._scoped_tools.values()):
            for tool_name in [n for n, e in scope_map.items() if e.toolset.startswith("mcp-")]:
                scope_map.pop(tool_name, None)

    _clear()
    try:
        yield
    finally:
        ss.set_multiplex_active(False)
        set_hermes_home_override(None)
        _clear()


def _scope(path):
    from hermes_constants import set_hermes_home_override, hermes_home_key
    set_hermes_home_override(str(path))
    return hermes_home_key(str(path))


def _tools_in_scope(server_name):
    from tools.registry import registry
    return registry.get_tool_names_for_toolset(f"mcp-{server_name}")


def test_select_new_servers_does_not_drop_divergent_route():
    """Route A already connected; discovering route B for the SAME name must NOT be deduped away."""
    from tools import mcp_tool as _core
    from tools.mcp_tool_discovery import _select_new_servers
    _core._servers[_core._server_key("cbm", CFG_A)] = _FakeServer("cbm", ["felix_only"], CFG_A)

    # B's key is absent -> B is a genuine new server, spawned on its own connection.
    assert _core._server_key("cbm", CFG_B) not in _core._servers
    picked = _select_new_servers({"cbm": CFG_B})
    assert "cbm" in picked, "divergent-route same-name server was wrongly dropped by name dedup"

    # Same route (A again) IS deduped: no second connection for an identical route.
    _core._server_connecting.clear()
    assert _select_new_servers({"cbm": CFG_A}) == {}


def test_each_profile_sees_only_its_own_route_tools(tmp_path):
    """Two live connections (A and B); each profile's heal resolves its OWN (name, fp)."""
    from tools import mcp_tool as _core
    from tools import mcp_tool_registration as reg
    _core._servers[_core._server_key("cbm", CFG_A)] = _FakeServer("cbm", ["felix_only"], CFG_A)
    _core._servers[_core._server_key("cbm", CFG_B)] = _FakeServer("cbm", ["jonas_only"], CFG_B)

    # Profile A resolves route A's connection.
    _scope(tmp_path / "A")
    assert reg.register_connected_into_current_scope({"cbm": CFG_A}) == 1
    got_a = _tools_in_scope("cbm")
    assert any("felix_only" in t for t in got_a), got_a
    assert not any("jonas_only" in t for t in got_a), got_a

    # Profile B resolves route B's connection — its own tools, never A's.
    _scope(tmp_path / "B")
    assert _tools_in_scope("cbm") == []  # B starts empty
    assert reg.register_connected_into_current_scope({"cbm": CFG_B}) == 1
    got_b = _tools_in_scope("cbm")
    assert any("jonas_only" in t for t in got_b), got_b
    assert not any("felix_only" in t for t in got_b), got_b


def test_lookup_server_fails_closed_on_foreign_route():
    """_lookup_server must NEVER return a foreign-route connection when this profile's own composite
    key is absent. Only a bare-string-keyed entry (no route info) is an acceptable fallback."""
    from tools import mcp_tool as _core
    # Only route A is live; a caller resolving route B (its own config) must get None, not A.
    _core._servers[_core._server_key("cbm", CFG_A)] = _FakeServer("cbm", ["a"], CFG_A)
    assert _core._lookup_server("cbm", CFG_B) is None          # foreign tuple route -> fail closed
    assert _core._lookup_server("cbm", CFG_A) is not None       # own route -> hit
    # A bare-string-keyed entry (directly-seeded / adopted w/o config) is the only allowed fallback.
    _core._servers.clear()
    _core._servers["cbm"] = _FakeServer("cbm", ["a"], CFG_A)
    assert _core._lookup_server("cbm", CFG_B) is not None       # lone bare entry -> permissive
    # But a bare entry PLUS a foreign tuple route must return the bare entry, never the foreign route.
    _core._servers[_core._server_key("cbm", CFG_A)] = _FakeServer("cbm", ["a"], CFG_A)
    assert _core._lookup_server("cbm", CFG_B) is _core._servers["cbm"]
