# PR 2572: rebase and merge assessment

Tracking: TASK-12089. PR: https://github.com/rmusser01/tldw_server/pull/2572.

## Recommendation

Keep this PR and merge it as an **experimental, documented compatibility subset** once current-head CI passes and the requester supplies the required human-written Change summary. The reviewed local implementation has no remaining confirmed blocking findings. This assessment does not establish full OpenAI client interoperability, live-provider latency, or production readiness.

Current `dev` has the design documents and existing native audio/chat services, but does not have this realtime protocol adapter, session orchestrator, or `/v1/realtime` route. The feature remains useful for clients that adopt its supported contract. It reuses the existing STT, chat, TTS, authentication, credential policy, transcript policy, and audio quota services.

## Rebase

- Original PR head: `a3b21f9faf3d39867d3050094cfdc1039276c528`.
- Rebased onto `origin/dev` at `40345571a2cfc8b3a8893545836097d27e4ee86c`.
- Backup: local branch `codex/pr2572-before-rebase-20260909`.
- Three already-applied design commits were skipped. Two documentation conflicts were resolved by retaining both the realtime additions and the current Voice Chat v1 section.
- Work was isolated in the existing PR worktree; the main checkout's unrelated changes were preserved.

## Confirmed findings addressed

| Finding | Resolution and regression coverage |
| --- | --- |
| Audio waited for the complete LLM response; slow clients accumulated adapter events | Interleave text and audio through a bounded pipeline queue; bound the WebSocket outbound queue. Tests hold the chat producer or socket sender and verify early audio/backpressure. Provider-internal buffering is not covered by this bound. |
| Cancellation left queued deltas, could deadlock buffered TTS cleanup, or leak an opened TTS session while credential usage was recorded | Discard cancelled response events, close TTS before awaiting producer shutdown, close acquired sessions on cancellation during usage recording. Tests block a real buffered-session provider and the credential operation. |
| A final event could be dequeued while incoming STT work was awaited, then silently discarded | Recheck the outbound task after processing the incoming command and observe generation completion in the wait set. A deterministic concurrent commit regression verifies delivery of `response.done`. |
| New routes bypassed audio concurrency/daily usage enforcement | Reserve and release the existing audio stream quota, renew its lease, and charge committed audio before STT. Denial, heartbeat, factory failure and disconnect tests cover the boundary. |
| Default chat/TTS integration bypassed scoped provider credentials and admin policy | Reuse the trusted request scope and existing credential runtime/context manager, retaining the snapshot for the stream lifetime. Tests cover credential propagation, denial, and cleanup. |
| Raw STT transcript bypassed the existing text policy | Resolve and apply the effective STT policy before session events, LLM input, or persistence see the transcript. Test uses the real redaction function. |
| Cookie authentication consumed the first protocol message before acceptance | Honor the existing no-initial-auth-message mode immediately after cookie authentication. Regression verifies the first `session.update` remains unread. |
| Follow-up turns lost preceding dialogue | Keep the last 20 completed user/assistant messages in the per-session pipeline and include them in subsequent chat calls. A two-turn test checks the next request. |
| Input append repeatedly copied the entire audio buffer | Use `bytearray.extend`, snapshot once on commit, and retain the input-size limit. Existing append/clear/commit tests cover behavior. |
| Integer/blank persistence IDs and mid-response metadata updates violated the adapter contract | Normalize integer/nonempty string IDs, reject bool/float/blank values, and snapshot the target at response start. Parameterized tests cover IDs and updates during deltas. |
| Capabilities promised persistence while production used a no-op adapter | Advertise ephemeral sessions and persistence unsupported by default; document the extension metadata and requirement for an authorized persistence adapter. |
| Failed or empty output produced misleading lifecycle events | Check buffered TTS errors before audio completion; ensure content parts are added before completion; share the audio/transcript content index and include final text/transcript content in completed items/responses. |
| Supported wire shapes and documentation disagreed | Accept nested PCM format objects and output voice, serialize output content types correctly, and fix the spec's `output_item.created` typo to `output_item.added`. Exact protocol tests cover the supported subset. |
| Review hygiene issues | Add contextual privacy-safe handler logs, factory docstrings, fake-pipeline annotations, and a checked design-task acceptance criterion. |

An independent review identified the governance integration gaps and subsequently the three cancellation/event-delivery defects above. Each defect was reproduced before its fix. The final independent pass reported no additional must-fix findings.

## Existing PR comment disposition

All 23 existing inline comments were examined. Duplicate findings were addressed together.

- Fixed: comments `3510390929`, `3510443393`, `3510443394`, `3510443396`, `3510443401`, `3510443403`, `3510443404`, `3510443407`, `3510454825`, `3510454826`, `3510454828`, `3510454831`, `3510454836`, `3510454838`, `3510454841`, `3510454844`, `3510454846`, `3510454848`, `3510454851`.
- Retained the opt-in live smoke and its markers (`3510443397`, `3510443398`, `3510443400`): the approved implementation plan explicitly calls for a separately gated real-provider smoke test. Normal regression tests use fakes; missing explicit smoke configuration is an intentional, documented skip rather than a hidden failing test. Multiple pytest markers are valid.
- Retained pre-accept authentication rejection (`3510454835`): sending a JSON WebSocket frame before acceptance violates the ASGI handshake contract. The realtime routes deliberately authenticate before acceptance, with regression coverage for that behavior. Existing accepted audio routes keep their error-frame behavior.
- The suggested numeric ID conversion was narrowed to actual integers: accepting floats or booleans would invent unsupported conversation identifiers.

## Validation

All commands used the project virtual environment from the isolated worktree.

| Check | Result |
| --- | --- |
| Rebased baseline realtime and route-policy suite | 110 passed, 1 intentional live-smoke skip |
| Final `tests/Audio/test_realtime_*.py` plus `tests/Resource_Governance/test_realtime_route_policy.py` | 140 passed, 1 intentional live-smoke skip; 24 warnings |
| Shared audio service, STT policy, TTS session sanitization, credential boundary, router import resilience, and realtime governance slice | 92 passed; 4 warnings |
| Ruff on realtime implementation/endpoints and tests, Python 3.11 target | Passed |
| Black on the same scope | Passed |
| Legacy `/complete` and HTTP client patching repository guards | Passed |
| Python compilation of touched realtime/audio/TTS implementation | Passed |
| Bandit across all PR production Python paths | No findings; no scan errors (5,282 lines analyzed) |
| `git diff --check` | Passed |

The local suites do not replace the repository's cross-platform CI matrix. Before this update, GitHub showed 22 failed, 743 successful and 4 skipped checks on the old PR head, plus one check without a conclusion. Failures included audio/ingestion shards, E2E gates, and full-suite aggregators. Those historical results neither validate nor diagnose the rebased head; its checks must be assessed separately.

## Explicit limits and merge conditions

- The endpoint uses manual turns, 16 kHz mono PCM16 input, 24 kHz output, and a fixed text-plus-audio response. Upstream OpenAI's PCM session contract requires 24 kHz input, among other differences. See the [official client-event reference](https://developers.openai.com/api/reference/resources/realtime/client-events) and the supported contract in `Docs/Audio_Streaming_Protocol.md`. A drop-in OpenAI GA client compatibility claim is inappropriate.
- Native incremental TTS can now deliver audio before the LLM finishes. Buffered providers still wait for buffered text. No live latency benchmark or provider interoperability result is claimed.
- WebRTC, automatic VAD, tool calls, truncation/barge-in parity, and durable production persistence remain outside this stage. Persistence metadata alone does not save data.
- Current-head required CI must pass before merging. This review does not merge the PR or silently waive failed checks.
- The requester must write their own **Change summary** explaining both the change and why the implementation choices were made, as required by `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`. This agent-written report does not satisfy that gate.

The temporary follow-up plan `IMPLEMENTATION_PLAN_pr2572_rebase_review.md` completed three stages: rebase/applicability, regression fixes, and verification/recommendation. Its outcome is retained here and in TASK-12089; the temporary file is removed on completion.

## Subsequent Qodo follow-up

At the requester's direction, the branch was rebased again onto `dev`
`456eafb7a603449722ba8db806071a5e2aa5e7d6`. This rebase was conflict-free; backup
branch `codex/pr2572-before-rebase-20260910` preserves the previous published head.

Qodo's updated review of `3c6957516a` marked all prior code defects resolved and
retained three live-smoke policy comments. This follow-up supersedes the earlier
decision to retain that pytest test: real-provider verification now lives in
`Helper_Scripts/Testing-related/realtime_speech_smoke.py`, explicitly invoked with
a spoken WAV against a configured server. `test_realtime_smoke_cli.py` uses a fake
transport, one unit marker, and no configuration-dependent skips. The production
realtime pipeline remains unchanged from the preceding review.

The manual command validates 16 kHz mono PCM16 input, caps it at 30 seconds, and
splits it into frames below the protocol limit. An independent review caught the
initial single-frame assumption; a seven-second recording reproduced the defect
before the chunking fix. Fake-transport tests cover the request sequence, completed
and failed responses, error events, auth requirements, WAV validation and chunking.

The `backend-required` failure on the preceding head was traced to the new
capabilities path missing from the OpenAPI fingerprint. Canonical export reproduced
the exact CI mismatch. The fingerprint now contains 2,087 paths and 3,163 schemas,
with SHA-256 `00409b322975045d73f60965d6786d699734f8576505597b145f46be36dd3102`.
Frontend types were regenerated and include the capabilities endpoint; these
generated type files are intentionally gitignored by the existing workflow.
The job also logged a non-gating mypy/NumPy stub compatibility error under its
existing `continue-on-error` type-check step; this is not the contract-drift failure.

Local follow-up verification: **150 focused tests passed, no skips**, OpenAPI drift
check passed, frontend types generated, Ruff/Black passed, repository guards and
compilation passed, and Bandit reported zero findings or errors. Live-provider
interoperability and latency remain unverified. Posted Qodo findings and required
checks on the new head must be assessed before requesting the merge decision.

### Final validation and latest-dev refresh

Qodo reviewed `d61c9b8573ed1e5827393ef9c03466a0c11cb377` and reported **zero bugs
and zero rule violations**. All 50 checks passed; 26 checks were intentionally
skipped by workflow selection. The backend unit smoke also passed locally
(403 tests), and the deployment-shaped startup smoke returned the canonical
`/health` response.

While those checks ran, PR 2613 merged into dev, advancing it to
`f0248aaa00047d2ffcc3bde295d9fbb8296add8a`. The PR was rebased onto that commit
without conflicts. A tree comparison against the fully green head confirms the
only inherited changes are three Backlog documents and the existing audio-download
regression test; production code, the manual smoke command, and the API fingerprint
are unchanged. Focused realtime/route tests plus the updated download test passed
again: **153 passed, no skips**. Bandit again reported zero findings or errors.

The refreshed commit identifiers trigger another CI/Qodo cycle. Merge remains
conditional on that head's checks, the requester-owned Change summary (including
the implementation rationale), and explicit merge approval. The temporary
`IMPLEMENTATION_PLAN_pr2572_qodo_followup.md` is removed after its implementation
and verification outcomes are retained here and in TASK-12089.
