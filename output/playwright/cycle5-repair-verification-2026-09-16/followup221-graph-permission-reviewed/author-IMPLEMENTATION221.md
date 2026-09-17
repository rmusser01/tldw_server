# UAT221 — Notes Graph permission state and pending-request revocation

TASK13260.159. **Refrozen after the independent lifetime finding; re-review and native acceptance pending.** Only the four files in owned-manifest.json are owned. Exact copies are in review-snapshot/, and owned.patch includes the new test against the retained original baseline. The original candidate and its report/patch/manifest are preserved under pre-lifetime-review/. No browser, runtime, model, task/tracker or git mutation was performed.

## Result and authority boundary

The actual service403 displays “Notes graph is unavailable for this account. Ask an administrator for access.” Graph-derived content, search, inspector and suggestions disappear. Automatic requests stop for the mounted denied authority; explicit Refresh graph remains available online and restores access with one new request after permission changes. Replacement authority, genuine loading, transient500/503 failure, and offline cached-graph behavior remain covered.

The independent reviewer identified a missing lifetime boundary in the first candidate: radius1 could still resolve200 after radius2 received403 and refill cleared private pages, which a new mount immediately displayed. The mounted denied view itself stayed hidden. A cache clear alone was insufficient; the exact counterexample is retained in the independent packet and permanently added here, including a variant that keeps the old observer mounted.

On a non-canceled403 the hook now cancels other queries under the exact Notes workspace authority prefix, excludes the reporting query, waits for cancellation settlement, then clears paginated data. `revert:false` prevents canceled cursor fetches from returning earlier data through their success continuation. The query context key identifies the reporting query; its AbortSignal prevents a canceled request's eventual403 from clearing a successful manual recovery or setting denial again. Explicit cursor expansion resolves null for the existing expected403 or an actual TanStack cancellation; unrelated503 errors still reject.

Installed query-core5.90.20 cancelQueries cancels retryers synchronously; a settled retryer ignores later transport fulfillment before Query.setData. Passing the query signal also permits TanStack's ordinary observer-lifetime cancellation. This is logical query-result revocation, not a claim that background-proxy HTTP transport is physically aborted. Other authority prefixes and unrelated cache domains are not canceled. Single-response cursor entries are not interpreted as paginated data. A fresh opening may revalidate; this does not add a permanent authorization latch or global capability system.

## Causal verification

- Native original supplied receipt: actual /notes/graph403 at08:21:15.186UTC with missing notes.graph.read; native-cause.json binds the inputs and safe excerpt.
- Original causal RED5fail/4controls, followed by recovery/authority RED2fail/8controls and cached-reopen RED2fail/10controls, are retained. Initial incorrect textbox lookup and ignored-file lint attempts are retained as harness limits, not causal evidence.
- First frozen12-case test suite against original production through nonmutating loader:8fail/4controls. First candidate89/8 passed but was independently blocked by the pending-request race; these results do not establish final acceptance.
- Independent exact race:1fail/12pass on the first frozen candidate. Reviewer receipt: .tmp/uat221-independent-20260917/late-success-full-red.log.
- Permanent lifetime RED before patch:4fail/13pass. Both late base/reopen variants, pending cursor retirement, and old403 after recovery fail. Unrelated-authority completion passes.
- Final18-case permission suite against the prior candidate through vitest.pre-lifetime.config.ts: **4fail/14pass**. No production source was replaced. The additional503 cursor control preserves unrelated error propagation.
- Current final suite: **95 passed / 8 files / zero skipped,4.16s**, lifetime-final-green.log. The18 permission cases plus77 existing hook, authority, layout, accessibility and Connections controls all pass. The existing positive pagination/scope-switch behaviors remain asserted without amendment.

Tests use the real Workspace, hook, validated graph service, QueryClient, toolbar, inspector, AntD and production English/i18next ICU resources. Only API transport and unavailable-jsdom Cytoscape canvas rendering are doubled. There was no test weakening or change to pre-existing tests. Root's separate224 graph-ID normalization service repair is outside this ownership scope.

## Static checks and limits

- Scoped ESLint: original baseline0errors/0warnings; final0/0 (eslint-lifetime-scoped.json). Existing root Next pages-directory stderr is retained.
- Full frontend compiler:90baseline/90current, identical diagnostic messages after line-position normalization. No owned-path diagnostic added; this is not a clean full build.
- Bandit via project venv:0findings and3TSXparse errors, so no meaningful TypeScript security coverage. Manual boundary review checks prefix-scoped cancellation, error classification, fresh manual recovery, unrelated-authority continuity, no raw diagnostic rendering, and no backend-policy or secret handling change.
- Whitespace passes. English assets/locale/en/option.json is the active source; generated public locales are untouched.
- Native denied/admin browser acceptance and independent re-review remain gates. No native or full-matrix success is claimed.

## Reproduce

From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.permission.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphAuthorityScope.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.loading-i18n.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.view-mode.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage27.source-links.test.tsx
bunx vitest run --config ../../.tmp/uat221-repair-20260917/vitest.baseline.config.ts ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.permission.test.tsx
bunx tsc --noEmit --incremental false --pretty false
```

The second command intentionally fails against the old sources. From repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx apps/packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.permission.test.tsx -f json
```
