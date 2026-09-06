"""Per-home MCP discovery: one discovery thread per HERMES_HOME profile (D5).

``start_background_mcp_discovery`` keys its started-set and thread-dict by
``hermes_home_key(get_hermes_home_override())`` so each profile spawns its own discovery under its
own home_override; the same profile twice reuses one thread. Single-profile behavior is unchanged
(the set/dict just have one member).
"""
from __future__ import annotations

import logging
import threading

import pytest

from hermes_cli import mcp_startup
from hermes_constants import hermes_home_key, set_hermes_home_override


@pytest.fixture(autouse=True)
def _reset_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    mcp_startup._mcp_discovery_started = set()
    mcp_startup._mcp_discovery_thread = {}
    try:
        yield
    finally:
        for t in list(mcp_startup._mcp_discovery_thread.values()):
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
        set_hermes_home_override(None)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread


def _install_blocking_discovery(monkeypatch, stop: threading.Event, calls: dict):
    def _discover() -> None:
        calls["n"] = calls.get("n", 0) + 1
        stop.wait()

    monkeypatch.setattr(mcp_startup, "_has_configured_mcp_servers", lambda: True)
    monkeypatch.setattr(mcp_startup, "_discover_mcp_tools_without_interactive_oauth", _discover)
    monkeypatch.setattr(mcp_startup, "_any_mcp_connected", lambda: True)


def test_discovery_spawns_one_thread_per_home(monkeypatch, tmp_path):
    stop = threading.Event()
    calls: dict = {}
    _install_blocking_discovery(monkeypatch, stop, calls)
    log = logging.getLogger("test")

    a, b = tmp_path / "A", tmp_path / "B"
    key_a, key_b = hermes_home_key(str(a)), hermes_home_key(str(b))
    try:
        set_hermes_home_override(str(a))
        mcp_startup.start_background_mcp_discovery(logger=log, thread_name="disc-a")
        set_hermes_home_override(str(b))
        mcp_startup.start_background_mcp_discovery(logger=log, thread_name="disc-b")

        assert mcp_startup._mcp_discovery_started == {key_a, key_b}
        assert set(mcp_startup._mcp_discovery_thread) == {key_a, key_b}
        assert mcp_startup._mcp_discovery_thread[key_a] is not mcp_startup._mcp_discovery_thread[key_b]
    finally:
        stop.set()


def test_same_home_twice_reuses_one_thread(monkeypatch, tmp_path):
    stop = threading.Event()
    calls: dict = {}
    _install_blocking_discovery(monkeypatch, stop, calls)
    log = logging.getLogger("test")

    a = tmp_path / "A"
    key_a = hermes_home_key(str(a))
    try:
        set_hermes_home_override(str(a))
        mcp_startup.start_background_mcp_discovery(logger=log, thread_name="disc-a")
        first = mcp_startup._mcp_discovery_thread[key_a]
        # Second call under the same home: the live thread is reused, no new slot.
        mcp_startup.start_background_mcp_discovery(logger=log, thread_name="disc-a")
        assert mcp_startup._mcp_discovery_started == {key_a}
        assert mcp_startup._mcp_discovery_thread[key_a] is first
    finally:
        stop.set()
