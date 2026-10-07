---
id: TASK-13413
title: Sandbox websocket multi-subscriber live stream test flakes (off-by-one frames)
status: Done
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-03 02:47'
labels:
  - bug
  - sandbox
  - testing
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_Server_API/tests/sandbox/test_ws_multi_subscribers.py::test_ws_multi_subs_live_stream failed in CI (platform-sandbox-ws-streams, #3058 run 36693231386, 2026-09-30): assert [1, 2, 3, 4] == [2, 3, 4, 5]. A subscriber saw a frame published before it attached, or missed the last one. Unrelated to #3058's audio route change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Subscription and replay ordering race identified and the test passes reliably
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CI failure mechanism (#3058, SHA 3223160834): the old sandbox conftest autouse fixture patched the global asyncio.sleep to 10ms, so the 10s endpoint heartbeat fired within the test, and the old test read 4 raw frames. ws1's handler published heartbeat seq 1 before ws2 subscribed. _publish_local never buffers heartbeats, so ws2 missed it: ws1=[hb1,start2,L1 3,L2 4], ws2=[start2,L1 3,L2 4,end5]. dev already fixed that (020e853e5a opt-in heartbeat clock, 6b56b08a65 data-frame reader). Product race found while reading the replay path: subscribe_with_buffer replayed frames that were still queued for dispatch, stamped their seq there, then registered the subscriber, so the dispatcher delivered them again (duplicates), and a queued heartbeat got a later seq than buffered frames published after it (existing subscribers saw seq 1,3,2). Endpoint evidence on dev: a mid-stream attach stress gave duplicate seqs in 3 of 5 runs, and the first WS client got pre-connect frames twice. Fix: subscribe dispatches pending frames to existing subscribers first, then replays the fully stamped buffer once and registers under the same lock; dispatch fans out under the lock. from_seq semantics unchanged (test_ws_resume_from_seq passes). Also found: test_ws_multi_subs_live_stream failed deterministically when run alone, because settings were cached with synthetic frames on; _client now calls clear_config_cache (pattern from test_ws_signed_validation). Verification: RED on original streams.py for the new hub test, the WS sentinel assertion and the mid-stream stress (3/5 fail). GREEN: WS file + hub file + stress 30/30; live test alone 10/10; platform-sandbox-ws-streams shard + redis fanout + ACP runner client 134 passed; docs contract 9 passed. Bandit -ll streams.py: no issues. Ruff: streams.py clean; 8 findings already in the test files, none on changed lines. Docs: Sandbox_API.md states replay and delivery guarantees. PR #3081.

Qodo follow-up on PR #3081 (commit d380ec64bf). The previous commit ran the full fan-out (deepcopy plus call_soon_threadsafe for each frame and subscriber) under the hub-wide RLock, in _do_dispatch and on subscribe, so one run's backlog stalled every other run. Fix: subscribe no longer dispatches. Under the lock it stamps seq on queued frames in publish order (heartbeats included), snapshots the buffer, and registers the subscriber with live_from = the next seq; replay copies happen after the lock is released. The dispatcher returns to the dev shape: pop, stamp and snapshot under the lock, then copy and deliver outside it, skipping subscribers whose live_from is above the frame's seq because they already replayed it. The two subscribe-with-buffer methods now share one implementation. New test_hub_slow_fanout_on_one_run_does_not_block_other_runs: a frame whose deepcopy blocks on an Event stalls one run's dispatcher, and subscribe plus publish on another run must finish within 2s. RED on 63739b3014, GREEN now. Both new hub tests are marked @pytest.mark.unit. Verification: test_ws_multi_subscribers + test_streams_hub_resume_and_ordering + the mid-stream attach stress passed 30/30. The platform-sandbox-ws-streams shard + test_redis_fanout + test_acp_sandbox_runner_client + the docs contract: 144 passed. Bandit -ll streams.py: no issues. Ruff: streams.py clean; the 8 test-file findings were already there before this PR and none is on a changed line.

Follow-up after #3081 merged (2026-10-03 02:44Z): drain_buffer still stamped buffered frames one at a time, the old out-of-order pattern, so a frame could take a seq ahead of an earlier queued heartbeat. It now calls _stamp_pending_locked first (follow-up PR chore/followups-13410-13416). New test_drain_buffer_numbers_frames_in_publish_order fails on the old code (stdout seq 1 instead of 2). Sandbox stream tests: 16 passed; ruff and Bandit clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The CI flake was an unbuffered heartbeat (made fast by the old global sleep patch) that the old test counted as data. Both causes were already fixed on dev. Fixed a real hub race in the product: a subscriber attaching between publish and dispatch got frames twice, and seq could be stamped out of publish order. Subscribe now stamps queued frames in publish order, replays the buffer once and registers with a live_from threshold. Fan-out runs outside the hub lock, so one run cannot stall others. Also fixed the live-stream test's settings-cache isolation (it failed when run alone) and added hub, WS and cross-run blocking regressions plus API doc guarantees. 30/30 runs pass. PR #3081.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
