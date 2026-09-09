# Buddy and Persona PR review follow-up

PR: [2933](https://github.com/rmusser01/tldw_server/pull/2933), targeting dev. Follow-up reviewed from `b7f5b39e0d`. The approved scope remains Buddy/Persona in the server and shared WebUI/extension.

ADR required: no new ADR. ADR path: `backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md`. These are corrections within the existing ownership, authorization and runtime contracts.

## Review dispositions

| PR comment | Outcome |
| --- | --- |
| 3963510012 | Moved list/get/Stop orchestration from the HTTP endpoint into existing Buddy core use cases. Stop still revokes SQL publication before local cancellation. |
| 3963510019 | Added a composed-application HTTP test for protected starter artwork: actual PNG bytes, media type, cache and nosniff headers, missing assets, and rejected authentication. |
| 3963510025 | Documented model purposes, field constraints and validators. Schema descriptions were regenerated into OpenAPI. |
| 3963510032 | Corrected lifespan generator annotation to AsyncIterator and documented the synchronous error context manager. |
| 3963510038 | Centralized Buddy exceptions without changing their identities or base classes. |
| 3963510046 | Missing provider/model now raises a specific configuration exception; unrelated implementation ValueError is no longer misclassified as client validation. |
| 3963510055 | Classified ledger tests as integration tests. |
| 3963510062 | Replaced a fixed delay with bounded waiting for the actual blocked provider and runtime worker to settle before checking stopped-output fencing. |
| 3963510065 | Kept established cross-feature integration tests in Persona: they exercise artwork, Buddy persistence, workspace ownership and Chat boundaries and already run in all five CI variants. A rename-only move would add fixture/shard churn without changing coverage. |
| 3963510070 | Checked workspace client ownership before title disclosure. Foreign and missing targets return the same 404 and leave attachment state unchanged. |
| 3963510073 | No redundant second character-store clear: the active character derives from canonical selectedAssistant. Strengthened the real canonical-character clear regression instead. |
| 3963510078 | Retargeting resets the displayed target. Deferred Apply work is also fenced by committed target generation; old work cannot continue writing or close/overwrite the new draft. Already committed artwork/default/attachment outcomes are reported accurately. |
| 3963510082 | Preserved the connected Persona while another Persona is edited. New connections already pass through disconnect/reset; a strengthened route regression verifies the artwork/session identity pairing through reconnection. |

An independent source review identified the asynchronous retarget race after the initial reset fix. Deferred-promise regressions failed for creation and workspace saving before correction. Final coverage also retargets during attachment and host refresh. A committed attachment still triggers the host's read-only refresh so its version is current, but cannot close the retargeted editor. A saved starter is retained for retry without duplicate publication.

## Focused verification

- Before this follow-up: fresh shared UI run 437 passed in 26 files; backend run 57 passed with one PostgreSQL-environment skip. Earlier real PostgreSQL 18 foundation run passed 34 with zero skips.
- Follow-up UI: 109 passed across management, Playground identity and Persona session suites. After the additional race fix, management/host suites passed 24 (10 management and 14 host). Counts overlap; they are not a combined unique total.
- Backend follow-up: 50 passed and one PostgreSQL skip in the affected three-file run, plus one incorrect HTTP500 test expectation. TestClient correctly rethrows the deliberate implementation error; after correcting that assertion, the remaining case passed separately. All 51 runnable cases therefore passed across the two runs on unchanged backend production source.
- Twelve affected Python files pass Ruff and its formatter. The large Chat endpoint retains the same three pre-existing import diagnostics and formatting debt; only its local exception import changed. Bandit reports zero findings. Four touched UI files already had whole-file formatting debt; no clean whole-file formatting claim is made.
- Generated documentation: 33 focused tests passed after refreshing Docs/Published.
- OpenAPI exporter and installed type generator succeeded; subsequent fingerprint drift check passed (2,085 paths / 3,162 schemas).
- Buddy CommandPalette fallback fix: 11 failures before correction; 26 passed across the four affected suites afterward. Focused pinned ESLint reports zero errors.
- Workspace creation test registered in all five applicable CI shard variants; coverage guard reports zero newly uncovered files.
- Independent source review confirmed the async retarget fix with no remaining Important or Critical findings; the reviewer did not rerun tests. Whitespace verification passes. No full repository suite was run.

## Remaining CI and environment limits

The older published head failed some frontend shards in addition to the generated docs/API and command-palette failures corrected here. Remaining failures involve Playground handoff mocks, Quiz export mocks, handoff locale expectations and Research stage3 job expectations. The affected tests and handoff hook are unchanged from dev; this is source comparison, not a pristine-dev test run, and does not prove every failure unrelated. The VisualPackEditor copied-draft case passes its exact isolated rerun (369ms) and previously passed in the full 65-case file; its CI timing/environment failure has no reproduced root cause. Follow the new PR checks for their current status.

The follow-up used composed HTTP authentication dependencies and controlled provider responses. No real token/session browser smoke, model-provider round trip, microphone/audio, packaged extension launch, or new PostgreSQL service run was performed. Existing runtime limitations remain: process-owned accepted Buddy work ends on restart without replay or persisted credentials; multi-worker acceptance requires principal affinity. Legacy Persona Live remains connection-owned.
