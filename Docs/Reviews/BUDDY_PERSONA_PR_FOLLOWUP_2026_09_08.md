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

## Prior-head CI and environment limits

The older published head failed some frontend shards in addition to the generated docs/API and command-palette failures corrected here. Remaining failures involve Playground handoff mocks, Quiz export mocks, handoff locale expectations and Research stage3 job expectations. The affected tests and handoff hook are unchanged from dev; this is source comparison, not a pristine-dev test run, and does not prove every failure unrelated. The VisualPackEditor copied-draft case passes its exact isolated rerun (369ms) and previously passed in the full 65-case file; its CI timing/environment failure has no reproduced root cause. Follow the new PR checks for their current status.

The follow-up used composed HTTP authentication dependencies and controlled provider responses. No real token/session browser smoke, model-provider round trip, microphone/audio, packaged extension launch, or new PostgreSQL service run was performed. Existing runtime limitations remain: process-owned accepted Buddy work ends on restart without replay or persisted credentials; multi-worker acceptance requires principal affinity. Legacy Persona Live remains connection-owned.

## September 9 frontend CI repair

Dev remains `6cd2745f696af04668a61c20b84ab8a9e69ca5e4`; rebasing the feature branch was a no-op. The remaining published frontend shard failures were reproduced and corrected in six existing test files. No production code or dependencies changed during this repair.

- Playground tests now supply the router, provider, MCP and service-prompt contracts their real consumers use. The composer mock honors external send controls, the submit mock returns its discriminated result, and the diagnostics test clicks the recovery button and verifies its exact route. Existing refinement, invalidation, metadata and follow-up assertions remain.
- Quiz export tests provide the currently required remediation hooks without changing export/filter expectations.
- Research tests follow the documented parent-upsert publication boundary from `9ffb7f3f08`, keep projection closed on failed upsert, reset unused one-shot mock queues, and use deferred results to verify transient failure recovery versus repeated failure. Exact error-state and call-count assertions remain.
- Sidepanel copy tests distinguish route-only navigation from explicit draft/context transfer, matching the intentional copy change in `a171773d665`.

Fresh focused verification: voice **2 passed**, image refinement **15 passed** with no unhandled errors, and the four remaining affected files **41 passed**. These are 58 tests across six files; no skips or behavior assertion removals. A separate unchanged-backend run passed **60 tests with one PostgreSQL-environment skip** across independent Buddies, turn ledger, turn acceptance and workspace assistant creation. No full local suite or real provider/audio run was performed. Published CI on the new head remains the merge gate; these focused results do not substitute for it.

After changed-range formatting and replacing three new broad test annotations, the combined six-file run passed all **58 tests in12.86s**. Pinned ESLint reports zero errors and zero introduced warnings compared with HEAD. The extension formatter has no remaining changes on edited ranges; existing whole-file formatting debt remains.

Independent final source review approved the six-file CI repair with no Critical, Important or minor findings. The reviewer checked the actual consumer contracts and cited history, including Research publication/failure gates and polling generations, the refinement prompt/submit contracts, diagnostics navigation and separate locale copy. Review used the supplied test evidence without rerunning suites.

## Final shard7 synchronization correction

After the six-file repair, CI exposed one timing-sensitive ChatPane diagnostics-card test (1failed/140passed in its141-case shard). Both CI exact-base replays passed; this is not claimed as a baseline failure. A controlled deferred diagnostics response reproduced the same failure locally: observing request dispatch did not guarantee React had published the cards. The test now verifies loading and zero cards, releases the response, awaits rendered output, and requires exactly8cards plus the120-turn summary. Existing successful and forbidden-response checks remain unchanged.

The final affected file passes3tests; the related6-file ChatPane gate passes70tests. Pinned ESLint adds no errors or warnings, and no formatter changes intersect the edited ranges. No production code, timeout, retry, sleep, dependency or schema change was needed.

Independent review approved the final synchronization correction with no actionable findings, confirming actual loading-to-rendered-result synchronization, the stronger eight-of120 bound, and unchanged forbidden-response coverage.

## Shard 5 mock and conflict-reload corrections

CI on `9d3c76a62e` exposed two remaining test failures. The cockpit suite failed during import because its full service mock omitted `LEGACY_SERVICE_PROMPT_DEFAULTS`, which the real title service consumes. The same failure occurred in the exact-base replay. An asynchronous partial mock now preserves the actual service exports and overrides only the test-controlled model fetch.

The ReviewTab conflict-reload test passed the exact-base replay but failed on the PR head under CI timing. Its existing eventual Retry-absence assertion was correct; the earlier click could occur while automatic conflict recovery was still loading. Ant Design suppresses that click, so no manual reload starts. A deferred automatic response reproduced the ignored click and retained Retry deterministically. The repaired test explicitly checks that early click is ignored, waits for the loading guard to clear, then starts a separately deferred manual reload. It requires a second refetch, retains Retry while the manual response is pending, and removes Retry only after completion. The exact one-mutation assertion and neighboring conflict coverage remain.

Focused verification passed all **53 affected tests**, **72 tests including the direct title-service consumer**, and a final **42 cockpit tests** after a formatting adjustment. These counts overlap. Pinned ESLint adds no errors or warnings; edited ranges conform to the pinned formatter while unrelated whole-file debt remains. Whitespace checks pass. The two corrections change only existing test files, with no production, dependency, timeout, retry or Flashcards feature change.

Independent review approved both corrections with no actionable findings. The reviewer checked the installed Ant Design implementation and confirmed that its loading class and click guard share the same state. Existing ADR-005 remains applicable; no new architectural decision was needed. Published checks on the final head remain the merge gate.
