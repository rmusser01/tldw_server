# Current configured chat runtime — TASK-13596

The current Full WebUI completes ordinary saved conversation creation/reopen and character greeting/first reply/cold reopen using the same owned profile and one existing headless tab. Original nine conversations and 28 canonical messages remain exact. This is configured-profile functional evidence with a deterministic downstream provider, not first-install or real-model role-play quality acceptance.

Fresh `origin/dev` is a744577356ecd4db2d101689b5564a7a7c3c3f5a and is an ancestor of current1173e3e4ad. The latter includes [the shared greeting model-intent correction](CHAT_GREETING_MODEL_INTENT_13597_2026_10_12.md). ADR assessment: no new ADR; existing ADR049/050/059 and approved NN/G/H2 parent plans govern this runtime verification. No source or configuration change belongs to this task.

## Qualified runtime

Normal Node26 Next production webpack build runs with the original `development` build profile and `advanced` deployment mode. Production build mode and the house build-profile setting are separate. Build, shared-token verification and budget commands each exit0; shared593.3KB/600KB and chat795.6KB/900KB. The actual served manifest `LR5UsmKYlcp2mxLQ9_2cg` matches this artifact, verified Next78083 and supported3.12 API27579. Critical browser API traffic reaches original19767 with successful health, profile, history-selection and settings responses. Ambient19764 is untouched.

Verified old owned children retire gracefully before replacements; actual exits are checked before parent-manager notification. The API needed more than the initial50.43-second observation interval; that interval is not its total shutdown duration. No forced kill, private profile/key reset, authentication reinitialization or new browser/context/tab/window occurs. Current owned preview is19769; backend and downstream mock remain19767/19768.

Normal unchanged visible Server settings fields were exercised with Test connection and Save, followed by Return to chat. Actual health/storage checks succeed on19767. This checks configured setup/recovery, not a new account or first-install setup.

## Actual journeys

| Journey | Observed result | Boundary |
| --- | --- | --- |
| Find existing conversation | Known synthetic Finder baseline opens with two visible articles. | Read-only original case; all originals preserved afterward. |
| New ordinary conversation | UI create201/completion200; exact user input and assistant-parent link; two canonical rows and blank composer. Cold reopen retains both IDs. | Deterministic downstream response. |
| Existing character cold reopen | Prior case92fce296-aeab-4142-98da-0e042d528037 reopens in current artifact with its original three IDs and one opening. | Distinct from the new post-fix case below. |
| Current greeting and first reply | Unmodified Default Assistant card; Start with this greeting creates one opening. Actual protected Roleplay message201 and complete-v2 200 produce three canonical rows with exact user input and parent linkage. | New post-fix case d9d228ea-19e2-4eac-8e43-af4d78bcbbc9. |
| Current character cold reopen | New post-fix case reopens with three visible articles; full canonical rows deep-equal before/after reload, opening exactly once, composer blank. | Current manifest; one owned tab. |
| Recover drafts | Button top445.89/bottom473.89 inside scroller445.89–900 at scrollTop0; keyboard focus/Enter and pointer each open its dialog, then Escape closes it. | No draft restored, copied or deleted; no physical assistive-technology claim. |

Current screenshot: `output/playwright/chat-current-runtime-13596-postfix-roleplay.png`, inspected by root and both independent reviewers. Recover drafts is completely visible. Earlier partial-label screenshot reflected a scrolled composer position; source, bounds and interaction controls do not establish a clipping regression. Automatic character title is visible after current cold reopen; the earlier Untitled screenshot alone did not establish a title defect.

Immutable original baseline remains read-only. Before/after checks compare actual metadata-bearing raw canonical responses for all nine conversations/28 rows: deepEqual=true; baselineRewritten=false. New synthetic chats are separate from these originals. No canonical transcript, credential, header or startup diagnostic is published.

## Causal findings and review

The greeting guard root was separately fixed in TASK-13597, with four real same-value/away-back model-setter negatives and unchanged held positives. Its normal connected WebUI781/781, both application type projects, matched source/test lint and independent source reviews passed before this artifact was built.

Both reviewers give bounded current-runtime CLEAR. A character-status copy ambiguity remains separately tracked as TASK-13598: the client compatibility chip always says Selected — not verified, even after this successful saved flow. It is selection prediction, not proof of failed persistence. The correction must describe selection plainly; no saved-binding or actual prompt-use authority may be invented. This follows [NN/G visibility of system status](https://www.nngroup.com/articles/visibility-system-status/) and [familiar terminology in the ten heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/).

Excluded review-harness diagnostics: forcing packaged-production/quickstart profile baked default8000 rewrites and did not qualify this owned development runtime; an in-place Server-settings modal was incorrectly awaited as navigation; exact button text Send was initially used where the actual accessible name is Send message; exact Online text was absent while the observed model control reports Healthy. Correct actual profile/DOM controls resolved these without source/config/key resets or duplicate sends. None is evidence of a product regression or a qualified successful attempt.

Private evidence uses existing0600 files under `/private/tmp/chat-current-runtime-13596-*` and current build evidence under `/private/tmp/chat-greeting-revision-13597-*`. Post-fix and prior-character proof IDs remain separate. Normal exact-owned hooks and incremental local commit close this bounded task. Bandit is N/A for runtime/document-only scope; no Python code changed.

## Remaining acceptance

Master TASK-13458 and public H2 remain In Progress. Retention/reference roles, rich fork/read/send/frozen character/generated exact revision, client/Sync and original import/view-only/local-cache cases remain open. First-install, real-provider personality quality, Compact/installed extension, all viewport/theme/200% states, physical PostgreSQL/AT, interrupted/error journeys and participant study are not accepted by these happy-path proofs. The ten-heuristic whole-product assessment is not declared complete.
