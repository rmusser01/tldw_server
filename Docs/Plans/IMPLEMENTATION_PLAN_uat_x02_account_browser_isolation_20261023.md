# X02 account and browser isolation implementation plan

> Execute inline using the executing-plans workflow; reuse the authorized read-only reviewer.

**Goal:** Finish selected X02 before X03, proving reciprocal ownership and browser-state isolation with working owner controls.
**Architecture:** Use the qualified current3817-file installed API and unchanged TH frontend on private18908/18989, alongside the protected original runtime. Reuse official PostgreSQL isolation and existing lifecycle/receipt/session restoration helpers. Use supported real API/UI paths; mocks do not qualify live isolation.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md X-02 and apps/tldw-frontend/scripts/live-tier-uat/release-catalog-data.mjs X-02.
**Task:** TASK-13260.278.18.83.45.
**ADR assessment:** No new architecture rule for validation; existing AuthNZ/tenant ownership, PostgreSQL RLS and documented revocation contracts govern. Any demonstrated repair gets its own assessment and causal red/green before product edits.

## Global constraints

- Product source310642 plus documentation closure5c2e44 on dev22c9; current package3817 files, buildTHttd0p00MlSSxPE1Yyre. Refresh dev before later publication, not after every step.
- Preserve original API50010/49993, frontend50863/50855, Chrome54716/54710, five saved messages, exact draft/URL and full canonical objects. No original runtime cycle.
- One temporary independent browser context is necessary for isolation; keep the existing context/tab and close only the temporary one after validation. Never treat two shared-storage tabs as independent identities.
-600s per workflow/run/build,120s startup/child,180s inference,300s pytest case,15s operation/graceful cleanup; no force or assertion weakening. Reassess after three failures; latest human authorization covers necessary diagnostics/retries without another approval question.
- No stored secrets in output, model/cache/proof recovery, raw operator SQL or homemade PostgreSQL setup. Actual official managed fixture provisioning only.
- Keep failed outcomes failed. Dispose completed private payloads/operators promptly, retaining current artifacts and original restoration dependencies. Heartbeat remains paused.

## Stage1: Prepare current fixtures and owner controls
**Goal:** Fresh SQLite and official PostgreSQL multi-user profiles with ordinary Alice/Bob and Admin.
**Success Criteria:** Both ordinary users can create/read/update their source, Note, Chat, card, draft/handoff and Job. Same display names use distinct owner-scoped identities and private canary contents; numeric IDs may repeat across per-user stores. Admin is used only for setup/authorized operational controls. All stores and role bindings are explicit.
**Tests:** Qualified environment/source/package/build/owner/listener guards; normal initializer and real authenticated API creation. Any blocked resource blocks that isolation component.
**Status:** In Progress

- [ ] Read current endpoint/request/denial contracts and reuse existing profile/lifecycle helpers.
- [ ] Add minimal disposable BASE/x02_host.py and x02_api.py; guard their finite profile, process, role and deadline boundaries with small operator controls.
- [ ] Review before starting either private profile; record owned positives and resource IDs only in functional memory/private fixture.

## Stage2: Reciprocal API, retrieval and revocation
**Goal:** All ordinary-user directions reject foreign work while own work remains usable.
**Success Criteria:** Valid foreign IDs disclose no body/title/snippet/count/export/asset/job progress. Bob retrieves his private phrase; Alice retrieval candidates/citations/answer do not contain it without providing it. A real declared share/role grants only its scope; revocation follows the actual session/cache contract. Admin metadata never substitutes for ordinary controls.
**Tests:** Both backend cells; own read/write before each endpoint family, documented403/404 denials, export body/asset exclusions, Job ownership, same-name IDs, actual supported retrieval and representative real Chat.
**Status:** Not Started

- [ ] Run reviewed real clients; stop at a demonstrated failure and retain its exact contract/outcome.
- [ ] For a product defect, inspect every caller, add a normal causal regression, make the smallest shared repair, run affected checks plus compile/lint/scoped baseline-current Bandit, then independent review.
- [ ] Revalidate changed live behavior without replaying passing unrelated controls.

## Stage3: Browser switch and recovery
**Goal:** Account changes retire prior private UI state and in-flight work.
**Success Criteria:** Alice logout -> Bob login -> reload -> Back clears prior lists/drafts/handoffs/helpers/autocomplete/notifications. Switching during fetch/generation and an old tab cannot publish old-account results. Independent context remains isolated. Owned/foreign deep links, exports and background jobs retain positive/negative controls.
**Tests:** BASE/x02_browser.cjs with real login/logout/reload/Back and passive actual request receipts; current full canonical handback and zero-work graceful release. Screenshot for human UX review while remaining checks continue.
**Status:** Not Started

- [ ] Review browser driver and finite authority/write admission before action.
- [ ] Run sequential SQLite/PostgreSQL browser cells; use one temporary independent context and close it.
- [ ] Capture one settled screenshot; record functional and human UX results separately.

## Stage4: Review, dispose and close
**Goal:** Selected X02 accepted before X03.
**Success Criteria:** Ten catalog variants have explicit outcomes, original session and owners preserved, current task/master agree and independent actual/final review is clear. Completed private data/operators removed after release/ownership/open-handle guards.
**Tests:** Task normalization, scoped syntax/lint/security/diff checks and normal hooks/commits when product files change; no unchanged suite replay.
**Status:** Not Started

- [ ] Reconcile both multi-user cells and all recovery dispositions; single-user reciprocal cells remain N/A.
- [ ] Record concise results, retain failures/limitations, remove only this completed own plan, then activate X03.


## Confirmed handoff storage boundary defect

Before native startup, root and reviewer found that the default upload handoff
store uses the shared installation rather than configured USER_DB_BASE_DIR,
while ownership is numeric user ID alone. Separate deployments can reuse IDs;
known foreign draft IDs and profile-wide quotas therefore cross their boundary.
The cached factory also retains the first configured store.

ADR required:no; ADR020 governs configured database paths. Add normal real SQLite
get/delete regressions across two profile roots and a same-profile worker/quota
control before modifying production. The minimal repair uses DatabasePaths and
a factory cache keyed by its resolved file. Explicit paths, TTL, quotas and
cross-worker sharing remain; never import or fall back to the unscoped old DB.
Native API/browser work remains unexecuted until causal and affected checks,
security, review and refreshed packaging are qualified.

### Handoff repair verification

- Causal unchanged source:2 failed/1 passed; the known foreign draft was readable
  and deletable with the reused numeric actor. Corrected complete store and upload
  processing modules:18 passed in2.43s pytest/4.009s outer.
- Compile2passed. Configured Ruff:production0; tests retain exactly I001/BLE001.
  Black:production inherited flag unchanged, tests0. Bandit:production0, tests11
  identical inherited findings, no new findings/errors. Three new pytest asserts
  have narrow explained B101 annotations; import ordering/comments preserve behavior.
- Independent actual source, full added tests and affected callers: CLEAR.
- Pending: final staged review/normal commit and fresh3817-file private packaging
  before the first live fixture. Frontend unchanged; no rebuild is needed.
