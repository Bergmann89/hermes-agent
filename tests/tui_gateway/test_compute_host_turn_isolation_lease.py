"""Turn-isolation active-session lease handoff (Design 1).

With ``dashboard.turn_isolation`` on, a named-profile turn is admitted once in
the SERVING (dashboard) process -- it claims the per-session lease there -- and
then dispatched to a compute-host CHILD process to run. The child re-crosses the
ownership chokepoint in ``_run_prompt_submit`` (the #94778 double-writer guard),
but its rebuilt session dict does not carry the serving process's lease, and
``_is_same_writer`` can never match across a process boundary (it requires
``entry.pid == os.getpid()``). Without a handoff the child re-claims, sees the
serving process's live entry, and refuses the turn:

    Session ... already has a live owner (desktop, pid N, running 0m)

Design 1 hands the admission to the child as a disabled lease SENTINEL: the
serving process signals ``active_session_admitted`` in the turn frame, and the
child installs a disabled ``ActiveSessionLease`` so the chokepoint recheck is a
genuine no-op WITHOUT claiming a second registry slot and WITHOUT being able to
release the serving process's real lease. The single admission (the serving
process's) is preserved, so the double-writer guard is untouched.
"""

import threading

import pytest

from tui_gateway import server


def _session(**overrides):
    base = {
        "session_key": "sess-key",
        "history": [],
        "history_lock": threading.Lock(),
        "cols": 80,
    }
    base.update(overrides)
    return base


class TestTurnFrameAdmissionSignal:
    def test_frame_marks_admitted_when_serving_process_holds_lease(self):
        session = _session(active_session_lease=object())
        frame = server._compute_host_turn_frame("rid", "sid", session, "hello")
        assert frame["active_session_admitted"] is True

    def test_frame_admission_false_without_lease(self):
        session = _session()
        frame = server._compute_host_turn_frame("rid", "sid", session, "hello")
        assert frame["active_session_admitted"] is False


class TestChildDelegatedLease:
    def test_admitted_frame_makes_chokepoint_a_noop_without_reclaiming(
        self, monkeypatch
    ):
        # The child must NOT try to claim a second registry slot when the
        # serving process already admitted the turn.
        def _boom(*args, **kwargs):
            raise AssertionError(
                "child re-claimed the active-session slot despite admission"
            )

        monkeypatch.setattr(server, "_claim_active_session_slot", _boom)

        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )

        # The chokepoint recheck sees a lease and returns None (admitted),
        # never reaching the claim path above.
        assert server._ensure_active_session_slot("sid", session) is None

    def test_delegated_lease_release_is_a_noop(self, monkeypatch):
        # Releasing the child's sentinel must NOT touch the registry -- that
        # would drop the serving process's real lease out from under it.
        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )
        lease = session["active_session_lease"]

        import hermes_cli.active_sessions as active_sessions

        def _boom_release(_lease):
            raise AssertionError("delegated lease reached the registry release")

        # release() short-circuits on a disabled lease before calling the
        # module-level release_active_session, so this stays untouched.
        monkeypatch.setattr(active_sessions, "release_active_session", _boom_release)
        lease.release()  # must not raise

    def test_no_admission_installs_no_lease(self):
        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": False, "session_key": "sess-key"}
        )
        assert session.get("active_session_lease") is None

    def test_install_is_idempotent_and_preserves_a_real_lease(self):
        real = object()
        session = _session(active_session_lease=real)
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )
        # A real lease already present is never overwritten by the sentinel.
        assert session["active_session_lease"] is real

    def test_release_slot_on_sentinel_never_touches_registry(self, monkeypatch):
        # End-of-turn / finalize releases the session slot. For the delegated
        # sentinel this must be a clean no-op: the serving process still owns
        # the real registry entry.
        import hermes_cli.active_sessions as active_sessions

        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "sess-key"}
        )

        def _boom(_lease):
            raise AssertionError("delegated lease reached the registry release")

        monkeypatch.setattr(active_sessions, "release_active_session", _boom)
        assert server._release_active_session_slot(session) is True
        assert session.get("active_session_lease") is None

    def test_compression_transfer_on_sentinel_never_claims_a_second_slot(
        self, monkeypatch
    ):
        # Auto-compression rotates agent.session_id mid-turn, which fires
        # _transfer_active_session_slot in the CHILD. The sentinel must take
        # transfer_active_session's disabled-lease no-op branch -- never fall
        # through to _claim_active_session_slot (a real second registry entry
        # that would orphan the serving process's lease and break #94778).
        session = _session()
        server._install_delegated_active_session_lease(
            session, {"active_session_admitted": True, "session_key": "old-key"}
        )
        sentinel = session["active_session_lease"]

        def _boom_claim(*args, **kwargs):
            raise AssertionError(
                "compression transfer claimed a real second slot for a sentinel"
            )

        monkeypatch.setattr(server, "_claim_active_session_slot", _boom_claim)

        assert (
            server._transfer_active_session_slot(
                "sid", session, new_session_id="new-key"
            )
            is True
        )
        # Same sentinel object, re-anchored to the new id; no real claim.
        assert session["active_session_lease"] is sentinel
        assert sentinel.session_id == "new-key"

