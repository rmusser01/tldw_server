# UAT164 implementation handoff — TASK-13260.101

## Result and scope

Six owned files (three production, three existing test files) are frozen in owned-manifest.json and review-snapshot/. Source HEAD at handoff: ec28c34b7cf51f5e65100c0dba5f2bcd2b95711d. Root owns independent review, native acceptance, task updates, and commit. No browser, provider inference, service start, tracked documentation, or global tracker changes by this agent.

The successful metadata200/native no-ready-model case now shows truthful, surface-neutral guidance. Provider setup guide links directly to the existing local-provider configuration document, with explicit ask-administrator guidance. The external anchor uses target=_blank and rel=noopener noreferrer, preserving the current page. Home says Review model setup and describes readiness/setup guidance. It does not promise an editor in Model Settings. The full catalog is explicitly labeled Model catalog reference, including unconfigured/unavailable entries. Existing lower Configure server and Manage provider keys links remain.

Actual metadata401 shows Sign in to load models. Other metadata failures retain the sanitized RecoveryCallout and Retry; they do not also show empty setup guidance. Successful empty Refresh is a success, not a credential error. Failed metadata Refresh reports failure through the existing sanitized notification path.

## Evidence and mechanism

Native retained pre-configuration evidence: .tmp/fresh-uat-recovery-20260916/pg-multi-model-events.txt, pg-multi-model-settings.txt and pg-multi-home-ready.txt. Metadata returned200 with201 entries:199 provider_not_configured references, one configured but unsupported/unavailable MLX entry, and an image entry. No selectable chat models does not mean every provider is missing. Auth succeeded; provider-key/OAuth403 represented separate policy-disabled features.

fetchChatModels can intentionally return an empty cached fallback on request failure. Therefore Models now owns the already-existing tldw-providers-models metadata query as well as its existing chat-model discovery query. Query key, metadata normalization, and API endpoint are unchanged. The catalog renderer receives the query result; it creates no second query observer. Standalone AvailableModelsList retains its existing connected wrapper. This keeps a single metadata request and prevents the remount retry loop found in the first development GREEN attempt. The 401/503 controls assert one metadata request after the recovery state settles.

No auth-mode query, permission change, provider editor, new framework, or server write was introduced. The guide documents operator config and local provider readiness without depending on setup status. The previous /setup destination was wrong for completed/skipped multi-user setup: reviewer route probes proved that branch exposes a single-user API-key connection form. That route is unchanged; the CTA now bypasses it. The copy is deliberately valid for either authentication mode; the render controls vary enabled account-key responses versus multi-user policy-disabled403 responses. They assert the actual guide destination and external-anchor behavior, and do not claim to simulate a whole login.

## Validation

- red.log: actual ModelsBody + real catalog renderer and Home renders, external model/key requests mocked. Before production changes:6 expected failures /7 positive controls. Missing truthful empty guidance, misleading Home CTA, incorrect401 title, misleading failure/empty overlap, and empty Refresh behavior were exposed.
- final-green.log:31 tests /6 suites PASS. Includes existing Models display/helper/sanitization controls, Home options/sidepanel controls, and existing MultiUserExitPanel test. New cases cover catalog-only + configured-unavailable entries under both account-key policies,401/503 without empty guidance, bounded request count, pending metadata, failed Refresh→Retry recovery, successful empty Refresh, and empty→ready Refresh.
- eslint.json:6 owned paths,0 errors /0 warnings. Repo-root Next pages-path notice remains a tooling notice. Root bunx eslint first attempted package resolution and was blocked from its temporary directory; direct installed eslint entry point then succeeded, no install or dependency change.
- typecheck-final.log: final compiler replay exit2,90 existing diagnostics. typecheck-comparison.json compares retained UAT163 baseline by filename/code/message (line positions normalized):90 baseline /90 current,0 added /0 removed. This is comparison to a retained baseline, not a new baseline replay. No diagnostics on owned paths.
- bandit.json/log: requested Python Bandit command was run in the project venv against the3 production TSX files. It reports3 AST parse errors and no analyzed TypeScript security coverage. Do not present this as a clean TS security scan. Manual review found no new credential handling, external writes, permissions, or raw diagnostic exposure; existing sanitizer remains.
- git diff --check on the6 owned files:PASS.

Final test command (cwd apps/tldw-frontend):
`bunx vitest run ../packages/ui/src/components/Option/Models/__tests__ ../packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.providers.test.tsx ../packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomeShell.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/MultiUserExitPanel.test.tsx`

Typecheck: `bunx tsc --noEmit --incremental false --pretty false` (same cwd).

Lint (repo root): `node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs <six owned paths> -f json`.

Bandit (repo root): `source .venv/bin/activate && python -m bandit <three production paths> -f json -o .tmp/uat164-repair-20260916/bandit.json`.

## Limits and remaining acceptance

Native empty-to-configured and authenticated single/multi-user setup-route checks belong to root. No native claim from these render tests. The prior fresh-only /setup evidence did not cover completed/skipped setup and is not sufficient route proof. Metadata GET success is the available catalog health boundary; underlying chat-model cached-fallback behavior is unchanged. Existing nonfatal AbortError behavior is preserved (catalog empty rather than fatal recovery). Existing query cache keys/ownership semantics are unchanged. No unrelated cache/auth redesign.

Models uses new translation keys with English fallbacks to avoid selecting stale old translations; Home follows its existing English-copy convention. No sweeping locale edits. Catalog-only entries remain visible as reference; no filtering or sorting behavior changed.

## Independent review correction

The reviewer confirmed a P2 wrong-destination case with the real OptionSetup route in completed/skipped multi-user states (reviewer-setup-probe.txt, reviewer-probes.log). The earlier route assumption was based only on fresh setup after clicking Set up in WebUI. The correction is limited to the new CTA/copy and its render tests. Existing query, error, refresh and Home changes are retained. The target is the existing Docs/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md, using the same repository docs URL pattern as MultiUserExitPanel. Its opening sections document provider config and readiness diagnosis. No router edits or auth-mode query.

review-correction-red.log:4 failed /8 passed after changing render expectations before product edits. final-green.log:31/6 PASS after correction. Lint0/0 and Bandit3 TSX parse limitations reconfirmed. Earlier packet retained under pre-review-correction/. Final TypeScript replay completed: exit2,90 baseline/90 current,0 added/0 removed (typecheck-final.log and refreshed typecheck-comparison.json).
