# Chat Workspace No-Mock UAT, 2026-09-29

Historical baseline: the evidence below was collected before integration with
dev `60006a2fed2532d900d27accbc2cda87cb08c24b`. It does not certify the latest
candidate. TASK references below use the preserved pre-update namespace and
must not be confused with unrelated tasks now occupying those IDs upstream.
Current integration and new acceptance tracking: TASK-13396 and
`IMPLEMENTATION_PLAN_chat_workspace_latest_dev_repairs_2026_09_29.md`.
The SSE header repair is now authorized; implementation and fresh verification
are in progress. The old profile and evidence remain preserved.

Status: In progress, not accepted. Human approved the isolated-profile key save
on 2026-09-29, and normal browser authentication restoration succeeded. Live UAT
now identifies same-origin SSE compression buffering and missing conversation
restoration. The bounded SSE fix awaits explicit design approval. No
authentication, readiness, moderation or provider response was bypassed or
substituted.

Tracking: TASK-13255.8.1, TASK-13255.8.1.1, TASK-13255.16, TASK-13396,
TASK-13255.3.3, TASK-13255.12 and TASK-13255.15.

## Actual Runtime

- Worktree: `.worktrees/chat-workspace-a11y`, branch `codex/chat-workspace-a11y`.
- Isolated profile: `/private/tmp/chat-workspace-real-uat-20260929`.
- Next WebUI: `http://127.0.0.1:18089`; authenticated FastAPI: port 8000.
- Actual llama.cpp/Gemma GGUF: port 9099; actual isolated Redis: port 16379.
- Real MiniLM embeddings and Chroma storage, not readiness stamps.
- Official single-user AuthNZ initialization; credentials excluded from evidence.
- Existing user data, historical records and unrelated services were untouched.

## Live Evidence

| Case | Result |
| --- | --- |
| API stopped, browser readiness failure and Retry after recovery | Passed |
| Actual local provider discovery, validation and save | Passed |
| Provider Continue with full GGUF model identifier | Passed after TASK-13396 fix |
| Actual Gemma first-chat generation and completed setup state | Passed; response `chatcmpl-mM2n2KYDu3pYJshAnZzPnuoJACNwCzDM` |
| Document ingest, chunking, real embeddings and Chroma storage | Passed; media 1, canonical source queryable |
| Actual source-restricted hybrid RAG and Gemma generation | Passed; 37 seconds and Mira Chen, two citations to media 1, no cache hit or errors |
| Actual provider outage after durable user commit | HTTP 502, exactly one identified user retained |
| Restart provider and retry original UUID twice | Two actual answers, one original user, both correct parent links |
| Same-text new submission | Distinct new user UUID, correct assistant parent |
| Conflicting content and retry after a later user | HTTP 409 without historical mutation |
| SSE user receipt independent of assistant receipt | Observed before assistant acknowledgement |
| Raw HTTP stream disconnect before completion | Durable user retained; no server partial assistant saved |
| Enabled command guard, actual JSON and streaming requests | Passed; `/time` and an unknown valid candidate each return 422 before persistence, no history rows added |

Recovery conversation: `d22541ed-1810-4c34-9ffe-4897aabe533a`.
Original user: `18e60273-8db7-4fbd-9476-8724a0c62b64`; new same-text user:
`fb8c6967-303d-4b79-9cfb-c8f7d18dc37d`. Five stored rows remained unchanged
after the later-turn rejection. Evidence: `durable-api-results.json` in the profile.
An evidence-parser assumption was corrected from `role` to the deployed `sender`
field; already-completed turns were neither replayed nor deleted.

Grounded answer: "The primary sensor samples once every 37 seconds, and the
trial is coordinated by Mira Chen." The actual hybrid RAG response reports
`generation_executed=true`, provider `llama.cpp`, the full configured GGUF model
identifier and two citations to the ingested source. Evidence:
`grounded-rag-results.json` in the profile. This API check does not certify
browser source staging, model selection or the workspace chat flow.

The first command-guard probe used `/durable-uat-unknown-macro`. Hyphens are
outside the existing command-name grammar, so this was literal text and Gemma
answered normally; the probe's 422 assertion failed. The five original recovery
rows remain version 1 with their original IDs, contents and parent links; the
literal probe added two new rows, retained rather than deleted. The corrected
probe used `/durable_uat_unknown_macro` in a separate conversation
`4b9ea5a4-f9b7-4387-bacf-922065974ca9`. All four actual JSON/streaming guard
cases returned 422, with zero persisted rows. The original command configuration
was restored and the real API was verified ready. Evidence:
`command-guard-results.json` in the profile. No production parser change was
needed; the failed harness input is not counted as a passing guard case.

The first short stream probe completed inside the existing 1024-character
moderation holdback and is not proof of an early interruption. A longer actual
response emitted content before completion, then the client disconnected.
Conversation `49b027b9-bc03-458d-a235-8c37ff40a81e` retained user
`cd6791f2-c7b2-4533-86f7-7c1ebd123713`, but no assistant partial. The existing
streaming handler deliberately skips assistant save on cancellation/error;
this does not verify the browser's separate local partial-persistence path.

## Browser Continuation After Key Approval

The key was parsed locally and saved through the normal WebUI form; it was not
printed or included in evidence. The authenticated route rendered. With an empty
local workspace store, it stayed at "Loading workspace context" until normal
Research Workspace navigation initialized a workspace. The known server
Workspaces manager Open limitation also reproduced: opening
`d80bdbd2-9042-4fbb-a87a-52e3b2f4c845` set the URL query but left New Research
active. These are not successful workspace-activation cases.

UAT continued through normal My Media selection, associating the actual ingested
memo with browser-created workspace `f923b622-f678-421e-9680-e76288f96da4`.
The real catalog selected the full Gemma GGUF identifier. No app state was seeded.

| Browser Case | Result |
| --- | --- |
| Authenticated Chat Workspace with initialized context | Passed; actual source ready and actual model selected |
| Filter matching/no-match, stage, unstage, clear and summary insertion | Passed; insertion clears staged context and populates draft without sending |
| Browse selection | Badge changed; actual source content did not open, existing TASK-13255.15 remains open |
| Ctrl+Enter staged-source question | Passed; actual answer says 37 seconds and Mira Chen |
| Canonical persistence after grounded answer | Passed; exactly question text, correct assistant parent, no retrieval/system rows |
| Same-text new turn and two immediate send shortcuts | Passed; distinct user ID and only one new user/assistant pair |
| Stop actual Gemma, submit, restart and Retry same model | Passed; failed user retained once, retry reused ID and preserved earlier rows |
| Switch model picker and retry through actually offline Ollama | Passed failure handling; same durable user reused, no diagnostic assistant persisted |
| Successful switch between two live models | Not verified; only Gemma is running |
| 390x844 and 320x568 layout and panel navigation | Passed inspected views; no horizontal overflow, long model path wraps |
| Keyboard Sources/Chat panel switching | Passed; visible focus and failed draft retained |
| Fresh-document restoration in same browser profile | Failed; workspace/source/model restore but conversation and failed draft are absent |
| Long response and Stop after visible partial output | Blocked; real proxy buffers output, browser startup watchdog fires before visible output |

Browser conversation: `b36189ee-e6a0-4a99-9d6b-7ee551e91376`.
Grounded user: `27efad14-b575-4d59-ba02-82a1dc657359`; assistant:
`c29d2178-152c-4ece-b0e7-99d4203ca636`. Identical new question used
`f977ca20-d1e7-4eac-99cd-44a563b51345`. Real outage user:
`b947ccc2-b56f-442c-a4c8-04735405efc2`; recovered assistant:
`d4084f36-4058-4cfe-9bf4-b6562a810e75`. Failed long-response and model-switch
attempts reuse `9f6f2cb3-a0da-4ce1-87fd-cc831d0052b7`. The last actual API
inspection shows four users and three assistants, with all earlier checkpoint
rows unchanged. Evidence: `browser-history-results.json` in the profile.

Real timing comparison: model first content 0.166 seconds; FastAPI 2.046 seconds;
same-origin Next proxy 9.755 seconds, effectively at its 9.756-second completion,
with `Content-Encoding: gzip`. Browser logs identify the separate long-response
failure as "Chat response timed out before any visible output arrived" at the
unchanged 10-second startup limit. Next's installed compression implementation
honors `Cache-Control: no-transform`; all five existing chat SSE branches use
`no-cache` only. TASK-13255.3.3 tracks the bounded header fix, awaiting approval.
Timeouts and moderation were not changed. Evidence: `stream-timing-results.json`
and `stream-timing-next-results.json` in the profile.

Console observations include Next development HMR warnings and expected handled
provider/stream-timeout warnings. No runtime overlay occurred in these checks.
Viewport override was reset and the extra fresh-document test tab was closed;
the original failed-turn tab remains intact.

Screenshots in the profile: `browser-grounded-answer.jpg`,
`browser-provider-outage.jpg`, `browser-stream-timeout.jpg`,
`browser-mobile-inspector.jpg`, `browser-mobile-chat-320.jpg`,
`browser-mobile-inspector-320.jpg`, and
`browser-fresh-document-history-missing.jpg`.

## Pending Browser Acceptance

- SSE header fix and fresh same-origin streaming/Stop acceptance.
- Stop and transport failure after actual partial output, partial metadata and
  pre-stream retry rollback.
- Conversation/draft restore after reload and workspace return.
- Captured generation settings under actual settings changes, target/principal
  scope changes and successful switching between two running models.
- Actual source preview, server-workspace activation and empty-store startup.

No mocked Playwright route/response suite counts as UAT. In particular,
`chat-workspace-live-backend.spec.ts` uses response mocks and was not counted.
Authentication screenshot `authentication-required.jpg` is the historical
pre-approval checkpoint, not the current blocker.

## Separate Regression Checks

- Parent: 297 frontend tests across 11 changed-flow suites passed.
- Latest frontend worker: 512 tests across 23 suites passed; scoped TypeScript
  zero diagnostics, ESLint zero errors with existing warnings.
- Parent: 125 setup API tests passed, including GGUF identities and unchanged
  secret, `.env`, provider and receipt rejection. Explicit secret-in-GGUF
  first-chat submission and restoration coverage was added after review.
- Backend followups: 31 focused HTTP, 193 unit and 48 API tests passed;
  compileall, test Ruff and production Bandit passed. Production Ruff reports
  exactly three unchanged baseline import-order findings.
- Parent: 59 focused durable/migration/API tests and four actual PostgreSQL
  fixture tests passed with no PostgreSQL skips, before command followups.
- Parent final Bandit: eight production Python files, zero findings/errors.
- Final independent targeted review found no concrete regressions in the retry
  rollback/settings, command/continuation and GGUF followup changes.
- Whitespace check passed. Both final independent reviewers found no concrete
  concerns in the narrow setup model-identifier changes.

No commits, staging, GitHub mutations or issue closures. Keep the tracker open.
