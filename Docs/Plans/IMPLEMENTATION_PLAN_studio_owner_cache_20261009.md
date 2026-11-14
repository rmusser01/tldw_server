# Studio owner-bound cache and selection

Parent: TASK-13260.278.18.83.45 (root owns task notes).

## Scope and ADR assessment

Repair the demonstrated cross-account Studio cached-content flash and retained foreign project selection. Use the canonical Prompt owner readiness boundary, existing QueryClient factory, and an instance of the existing Studio store for each mounted owner. Bind default-project reads/writes to that owner. Preserve APIs, local Prompt behavior, numeric/UUID contracts, and existing tab invalidations. No startup-template, shared authentication/request-policy, backend or runtime changes.

ADR required: no. This applies existing authenticated-owner isolation to ephemeral Studio state, without a new persistence, authorization, transport or public API rule. Reviewed Docs/ADR/README.md; related existing patterns are Settings Prompt Studio owner remount, Claims workspace authority remount, and Repo2Txt instance stores.

## Stage 1: Reproduce ownership leak
**Goal**: Causal mounted tests using actual Studio container, Projects tab, Zustand state and TanStack QueryClient.
**Success Criteria**: Show cached A content/selection leaking across B switch and unresolved ownership; exercise late list/default settlement with one shared outer QueryClient.
**Tests**: New Studio owner privacy suite; synthetic transport responses only.
**Status**: Complete

## Stage 2: Isolate Studio lifetime
**Goal**: Minimal owner-bound cache/store and captured-owner defaults.
**Success Criteria**: Unknown ownership renders no private workspace; A/B/ABA and old callbacks cannot repopulate successor state; same-owner rerender preserves selection; offline guidance preserved.
**Tests**: New causal tests plus existing navigation and affected standalone store consumers under installed Node20 UI/Web configurations.
**Status**: In Progress

## Stage 3: Review and native acceptance
**Goal**: Freeze source after focused checks and independent review.
**Success Criteria**: Scoped lint has no added findings; root combined types and independent review pass; root validates actual browser switch.
**Tests**: Root-owned configured types/build/native gate. TS-only Bandit is not applicable; no security-proof claim.
**Status**: In Progress

No proof artifacts or runtime actions. Results reported inline. Maximum three failed repair attempts before reassessment; no dependency/worker changes.

## Qualification and limits (source frozen)

- Causal new suite: 4 failed / 2 controls passed on unchanged production, natural exit 1. Root accepted the two-file instance-boundary design.
- First correction: 5 passed / 1 new fixture expectation failed because the URL legitimately retained the Prompts subtab and showed the no-project gate. Corrected that expectation to verify the gate then navigate Projects. No production change for this failure.
- Expanded changed suites: 27 passed / 1 older fixture expected-call mismatch (captured owner argument). Preserved the original payload assertion and added the explicit owner argument.
- Final shared UI four-suite command: 48 passed / 3 failed / 51 total, natural exit 1, 75.62s; all 9 new privacy cases, all 19 navigation cases, and evaluation provenance passed. Three standalone TestRunHistory preview controls failed.
- Root-authorized temporary HEAD-store baseline for exactly those three IDs: 1 passed / 2 failed / 12 filtered, natural exit 1; finally restored exact candidate store bytes. Matched current-store run: 0 passed / 3 failed / 12 filtered, natural exit 1. Failure boundaries differ (Open run versus Open prompt version); this does not establish identical baseline failure or clear adjacency. No guards/assertions changed; no further attempts pending reassessment.
- Final configured Web changed suites: 28 passed / 2 files, no skips, natural exit 0, 4.56s. No claim that all 51 adjacent cases passed.
- Scoped real root ESLint baseline/current identical: 1 existing no-this-alias error in the older navigation MockWebSocket constructor, 24 inherited warnings; zero added diagnostics. New test and store each 0/0. Four TypeScript syntax parses clean; semantic UI/Web types remain root-owned and pending. TS-only Bandit N/A.
- Commands: installed Node20 running configured Vitest with testTimeout=300000 and a spawnSync outer timeout=600000, default workers unchanged. UI paths: components/Option/Prompt/__tests__/StudioTabContainer.{owner-privacy,stage6-navigation}.test.tsx plus Studio/TestCases/__tests__/TestRunHistory.test.tsx and Studio/Evaluations/__tests__/EvaluationProvenance.test.tsx under src. Web run used only the two changed suites via ../packages/ui/src. Attribution filter: refuses an original-version preview after a cookie owner switch|shows missing/deleted original-version metadata honestly.
- New tests use real shared outer QueryClient, owner-local QueryClient, Zustand, container and Projects tab; canonical owner hook, services and design-system presentation are controlled. Prompts child is a selection probe. No native auth-event/HTTP/browser/all-CRUD qualification.
- Root task owns independent review, unresolved adjacent controls, final types and actual browser acceptance. Stage 2 and Stage 3 remain In Progress. No proof files, runtime actions or task edits were performed.

## Independent-review continuation

Root authorized two connected corrections before edits: external Prompt Quick Test must share the same owner-bound Studio selection instance; module-global apiSend GET coalescing must not deliver A's held response to B's new Studio cache. Extend only PromptBody's existing owner wrapper plus the Studio service's local scheduling wrapper (coalesce:false); leave generic transport, owner helpers and recipe policy unchanged. Prompt author confirmed both paths are free. Add real Prompt-to-mounted-Studio and real apiSend/request-core held-response causal controls before production. Existing three standalone TestRunHistory failures remain unattributed, with no further unchanged replay.

### Continuation freeze and verification

- Initial continuation setup attempt had an editor-script typo and a missing safeStorageSerde test mock; neither was a product RED. Corrected only fixtures, then causal real-boundary command reproduced 2 failures / 1 same-owner control pass / 80 filtered, natural exit 1.
- Minimal correction: PromptBody owns the canonical owner-keyed Studio store; nested Studio providers reuse it, while independently mounted Studio still owns its own store. Studio service requests alone disable module-global GET coalescing. Generic transport and all recipe/owner policy branches remain unchanged.
- Same three controls passed after correction. Affected configured UI: 31 passed / 4 files / 80 filtered, natural exit 0, 7.78s. After the test selector received a lint-only explicit type, its sole UI case passed again (80 filtered, 9.62s). Configured Web on final bytes: 31 passed / 4 files / 80 filtered, natural exit 0, 6.19s. These filters exclude the earlier local Prompt cases intentionally.
- Exact affected command: installed Node20 with configured Vitest run on StudioTabContainer.owner-privacy.test.tsx, StudioTabContainer.stage6-navigation.test.tsx, services/__tests__/prompt-studio.owner-coalescing.test.ts and PromptBody.search-pagination.test.tsx; filter: Studio private workspace ownership|StudioTabContainer stage 6|held A transport|same-owner Studio reads|real Prompt Quick Test handoff. UI paths begin src/; Web paths begin ../packages/ui/src/. Case cap 300000ms, outer spawnSync cap 600000ms, default workers unchanged.
- Real transport test uses actual apiSend/request-core and two real QueryClients with held synthetic fetch responses. Real handoff test mounts PromptBody, its action hook, Studio container and actual execute drawer; canonical owner and services are controlled. No live authentication or browser acceptance claimed.
- Final scoped ESLint across eight changed source/test paths: 1 inherited navigation-fixture error / 101 inherited warnings; no additions. The three continuation existing paths matched retained pre-unit counts exactly: Prompt index 0/33, Studio service 0/24, Prompt fixture 0/20. Both new tests and store 0/0. Eight syntax parses clean; semantic types remain root-owned. TS-only Bandit N/A.
- Existing Prompt test bodies preserved: all 72 literal it() bodies byte-identical, one new body added (parameterized cases account for runtime count 80 prior cases). No previous assertion removed. The separate TestRunHistory failures above remain unattributed and were not replayed.
- Stage 2 remains In Progress pending independent review/adjacent disposition; Stage 3 remains In Progress pending root types and native acceptance. No proof artifacts, task edits, runtime or Git actions.
