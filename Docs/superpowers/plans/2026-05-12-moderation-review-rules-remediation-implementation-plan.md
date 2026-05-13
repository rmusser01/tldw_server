# Moderation Review And Rules Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/moderation` the moderation review queue, move existing rule configuration to `/moderation/rules`, harden the current rules UI, and add a backend-owned, sanitized review contract before shipping production review decisions.

**Architecture:** Keep the existing shared WebUI/extension route pattern, but split the moderation surface into a review route and a rules route. Use the existing moderation service and FastAPI router as the rules foundation, add a new SQLite-backed review store/service under `tldw_Server_API/app/core/Moderation/`, and expose typed review endpoints through the existing `/api/v1/moderation` namespace with separate reviewer permissions. Build frontend review UI only against the backend contract or explicit E2E fixtures, never by inferring production review items from local tester history.

**Tech Stack:** FastAPI, Pydantic, SQLite with `configure_sqlite_connection`, pytest, Next.js pages, shared React route registry, React Router, Ant Design where already used, lucide-react, Vitest, Testing Library, Playwright/CDP.

---

## Source Inputs

- Approved spec: `Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md`
- Planning task: `TASK-305`
- Current route entry: `apps/tldw-frontend/pages/moderation-playground.tsx`
- Current shared route: `apps/packages/ui/src/routes/option-moderation-playground.tsx`
- Current extension route: `apps/tldw-frontend/extension/routes/option-moderation-playground.tsx`
- Current shell: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`
- Current backend router: `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- Current backend schemas: `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- Current moderation service: `tldw_Server_API/app/core/Moderation/moderation_service.py`

## Non-Goals

- Do not redesign unrelated pages.
- Do not replace the moderation policy engine.
- Do not make the frontend synthesize production review events from blocklist rows or test history.
- Do not expose raw moderated content in the UI unless a later backend permission explicitly authorizes it.
- Do not use Computer Use for browser verification; use Playwright/CDP.

## Stage Dependency Map

1. Stage 1 is required before all other route work.
2. Stages 2 and 3 can run after Stage 1 and do not require review backend support.
3. Stage 5 is blocked by Stage 4.
4. Stages 6 and 7 are blocked by the Stage 5 decision path.
5. Stage 8 starts early with route/rules fixtures, then expands as backend and review UI stages land.

## Backlog Tracking Before Implementation

Before editing implementation files, create one Backlog task per stage or per smaller reviewable slice. Link each task to this plan and `TASK-303`. Suggested task split:

- `Implement moderation route and naming foundation`
- `Harden moderation content rules UI`
- `Fix moderation accessibility and responsive blockers`
- `Implement moderation review backend contract`
- `Implement moderation review queue MVP`
- `Implement moderation audit and undo workflow`
- `Implement moderation power-user review controls`
- `Add moderation fixtures and regression coverage`

Each implementation task should record touched files, verification results, known skips, and whether Bandit was required.

---

## Stage 1: Route And Naming Foundation

**Goal:** Align route names with the product model: `/moderation` is review, `/moderation/rules` is configuration, and `/moderation-playground` is a compatibility redirect.

**Files:**
- Create: `apps/tldw-frontend/pages/moderation.tsx`
- Create: `apps/tldw-frontend/pages/moderation/rules.tsx`
- Modify: `apps/tldw-frontend/pages/moderation-playground.tsx`
- Create: `apps/packages/ui/src/routes/option-moderation-review.tsx`
- Create: `apps/packages/ui/src/routes/option-moderation-rules.tsx`
- Modify: `apps/packages/ui/src/routes/option-moderation-playground.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/index.ts`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/ModerationReviewShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`
- Create: `apps/tldw-frontend/extension/routes/option-moderation-review.tsx`
- Create: `apps/tldw-frontend/extension/routes/option-moderation-rules.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-moderation-playground.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-paths.ts`
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify: `apps/packages/ui/src/services/settings/ui-settings.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.moderation.test.ts`
- Modify: `apps/packages/ui/src/tutorials/definitions/moderation.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/option.json`
- Modify: `apps/packages/ui/src/assets/locale/en/tutorials.json`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json`
- Modify: `apps/packages/ui/src/public/_locales/en/tutorials.json`
- Modify: `apps/tldw-frontend/e2e/page-mapping.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/smoke.setup.ts`
- Rename or replace: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts`
- Test: `apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx`
- Test: `apps/tldw-frontend/__tests__/stage5-route-alias-contract.test.ts`

**Implementation Tasks:**

- [ ] Add route constants to `apps/packages/ui/src/routes/route-paths.ts`:

```ts
export const MODERATION_REVIEW_PATH = "/moderation"
export const MODERATION_RULES_PATH = "/moderation/rules"
export const MODERATION_PLAYGROUND_LEGACY_PATH = "/moderation-playground"
```

- [ ] Create `apps/packages/ui/src/components/Option/ModerationReview/ModerationReviewShell.tsx` with an honest first slice state: heading "Moderation Review", queue count placeholders, permission/backend states, and a secondary link to `/moderation/rules`. Keep production data empty until Stage 4 exists.

- [ ] Create `apps/packages/ui/src/components/Option/ModerationReview/index.ts` exporting `ModerationReviewShell` so route wrappers can import from `@/components/Option/ModerationReview`.

- [ ] Create shared review route wrapper `apps/packages/ui/src/routes/option-moderation-review.tsx`:

```tsx
import OptionLayout from "~/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ModerationReviewShell } from "@/components/Option/ModerationReview"

const OptionModerationReview = () => (
  <RouteErrorBoundary routeId="moderation-review" routeLabel="Moderation Review">
    <OptionLayout>
      <PageShell className="py-6" maxWidthClassName="max-w-7xl">
        <ModerationReviewShell />
      </PageShell>
    </OptionLayout>
  </RouteErrorBoundary>
)

export default OptionModerationReview
```

- [ ] Create shared rules route wrapper `apps/packages/ui/src/routes/option-moderation-rules.tsx` that renders the existing `ModerationPlayground` component but uses route id `moderation-rules` and label `Content Rules`.

- [ ] Update the existing rules shell visible copy in `ModerationPlaygroundShell.tsx` from "Moderation Playground" to "Content Rules" and make the subtitle about policies, blocklists, overrides, and testing. Keep the component/folder name for this stage to avoid a broad rename.

- [ ] Change `apps/packages/ui/src/routes/option-moderation-playground.tsx` to a React Router `<Navigate to="/moderation/rules" replace />` alias for extension/shared options routing.

- [ ] Add Next pages: `apps/tldw-frontend/pages/moderation.tsx` dynamic-imports `@/routes/option-moderation-review`; `apps/tldw-frontend/pages/moderation/rules.tsx` dynamic-imports `@/routes/option-moderation-rules`.

- [ ] Change `apps/tldw-frontend/pages/moderation-playground.tsx` to use `RouteRedirect` with `to="/moderation/rules"` and explicit moved-route copy.

- [ ] Add extension wrappers mirroring the shared wrappers. If the sidepanel cannot fit the full queue at this stage, pass a `compact` prop to `ModerationReviewShell` and show counts/filters plus "Open full review"; do not route `/moderation` to rules.

- [ ] Update both route registries to lazy-load `OptionModerationReview` and `OptionModerationRules`, map `/moderation` and `/moderation/rules`, and keep `/moderation-playground` as a redirect alias.

- [ ] Update header shortcuts and sidebar defaults: replace `moderation-playground` with stable ids `moderation-review` and `moderation-rules`. Keep a migration fallback that ignores old stored `moderation-playground` values or maps them to `moderation-rules`.

- [ ] Update settings nav so the visible entries are "Moderation Review" and "Content Rules"; remove "Playground" from user-facing nav.

- [ ] Update only English locale source files in this stage. Leave generated/translated locale parity to the repo's existing localization workflow unless tests require flattened `_locales` values.

- [ ] Update page inventory, smoke setup, page mapping, and E2E route expectations for both canonical routes and the legacy redirect.

**Tests:**

- [ ] Update or add Vitest coverage for route constants and settings nav:

Run: `bunx vitest run apps/packages/ui/src/components/Layouts/__tests__/settings-nav.moderation.test.ts apps/tldw-frontend/__tests__/stage5-route-alias-contract.test.ts apps/tldw-frontend/__tests__/navigation/route-redirect-component.test.tsx`

Expected: all tests pass; redirect contract includes `/moderation-playground` -> `/moderation/rules`.

- [ ] Run route smoke for canonical and legacy paths after local dev server is available:

Run: `npx playwright test apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-routes.spec.ts --project=chromium`

Expected: `/moderation` loads review state, `/moderation/rules` loads content rules, `/moderation-playground` redirects.

**Status:** Complete

---

## Stage 2: Content Rules Hardening

**Goal:** Make the existing rules configuration surface safe, explainable, and consistent with the backend lint contract.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/BlocklistStudioPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/PolicySettingsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/UserOverridesPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/TestSandboxPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/AdvancedPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/hooks/useBlocklist.ts`
- Modify: `apps/packages/ui/src/services/moderation.ts`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/moderation-utils.ts`
- Test: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/BlocklistStudioPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/TestSandboxPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/AdvancedPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/UserOverridesPanel.test.tsx`
- Test: `apps/packages/ui/src/services/__tests__/moderation.service.contract.test.ts`

**Implementation Tasks:**

- [ ] Extend `BlocklistManagedItem` in `apps/packages/ui/src/services/moderation.ts` to include optional lint metadata from Stage 2 response enrichment:

```ts
export interface BlocklistManagedItem {
  id: number
  line: string
  pattern_type?: "literal" | "regex" | "comment" | "empty"
  action?: "block" | "redact" | "warn"
  replacement?: string | null
  categories?: string[]
  sample?: string | null
  ok?: boolean
  warning?: string | null
  error?: string | null
}
```

- [ ] In `useBlocklist.loadManaged`, call `lintBlocklist({ lines })` after loading if the backend response does not yet include metadata. Merge lint rows by line index.

- [ ] Replace the current `line.trim()` parsing in `BlocklistStudioPanel.tsx` with a small pure helper, for example `normalizeManagedBlocklistRows(items)`, that treats `pattern_type: "comment"` and `"empty"` as non-active rows. Unit-test this helper directly.

- [ ] Add search, active-only toggle, filters for `pattern_type`, `action`, `category`, and sort by `line`, `action`, `category`, or `pattern_type`. Default to active-only while preserving a "show comments and blanks" control.

- [ ] Update status badges and counts to count only active `literal` or `regex` rows with `ok !== false`.

- [ ] Add raw replace preview state in `useBlocklist`: `pendingRawPreview`, `previewRawReplace()`, `confirmRawReplace()`, and `cancelRawReplace()`. Preview must run lint first and compare `rawBaseline` to current draft.

- [ ] Change `Save / Replace` in `BlocklistStudioPanel.tsx` to open a diff/lint confirmation modal. Disable confirmation when lint has invalid rows.

- [ ] Change `AdvancedPanel.tsx` blocklist upload to read the file, show summary plus lint results, and require confirmation before `saveRawText`. Keep download and upload in separate visual groups.

- [ ] Add a session undo path for managed delete and raw replace. Minimum acceptable implementation: keep the deleted line or replaced raw baseline in component state and show an undo toast/button until the next reload.

- [ ] Update `TestSandboxPanel.tsx` with a "Why this result?" block. Derive explanations from `ModerationTestResponse.effective`, `action`, `category`, and `sample`: engine disabled, phase disabled, PII disabled, no matching rule, matched rule, user override, or global fallback.

- [ ] Update `PolicySettingsPanel.tsx` copy so read-only policy values, runtime overrides, and persist-to-disk scope are visibly distinct.

- [ ] Strengthen `UserOverridesPanel.tsx` destructive confirmations for bulk delete and show the effective policy summary for the selected user before save.

**Tests:**

- [ ] Add unit tests proving blank lines and comments render as comment/empty rows and are excluded from active counts.
- [ ] Add tests for search/filter/sort controls and active-only default.
- [ ] Add tests that raw save and upload cannot call `updateBlocklist` without preview confirmation.
- [ ] Add tests for "Why this result?" explanations for disabled engine, disabled phase, no match, and matched rule.

Run: `bunx vitest run apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/BlocklistStudioPanel.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/TestSandboxPanel.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/AdvancedPanel.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/UserOverridesPanel.test.tsx apps/packages/ui/src/services/__tests__/moderation.service.contract.test.ts`

Expected: all tests pass and no raw replace service call occurs before preview confirmation.

**Status:** Complete

---

## Stage 3: Accessibility And Responsive Pass

**Goal:** Clear the audit blockers on labels, keyboard use, focus, and 390px overflow before layering the review workflow on top.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationContextBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/BlocklistStudioPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/TestSandboxPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/UserOverridesPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/PolicySettingsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/AdvancedPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/components/QuickTestInline.tsx`
- Test: `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlayground.accessibility.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-responsive.spec.ts`

**Implementation Tasks:**

- [x] Add `aria-label` to icon-only buttons in `ModerationContextBar.tsx`: quick test, reload, and active-user clear.

- [x] Replace visual-only labels with programmatic labels. Use stable ids via `React.useId()` and either `htmlFor`/`id` or `aria-labelledby` for every input, textarea, select, AntD Select, and hidden file input trigger.

- [x] Convert phase segmented controls in `TestSandboxPanel.tsx` to `role="radiogroup"` plus `role="radio"`/`aria-checked`, or use native radio semantics.

- [x] Add keyboard handling to tab bars if needed: arrow keys move between tabs, Enter/Space activates, focus remains visible.

- [x] Make wide tables scroll inside their containers with `overflow-x-auto`, `min-w-*`, and no page-level horizontal scroll. Verify `ModerationPlaygroundShell` hero, context bar, status badges, tabs, category chips, and modals at 390px.

- [x] Add focus management for confirmation modals and undo controls. After a delete/replace modal closes, return focus to the trigger or the next stable action.

- [x] Ensure inline errors are associated with the related input using `aria-describedby`, and dynamic result/error areas use `aria-live="polite"` where appropriate.

**Tests:**

- [x] Add Testing Library assertions for accessible names:

```ts
expect(screen.getByRole("button", { name: /quick test/i })).toBeInTheDocument()
expect(screen.getByLabelText(/sample text/i)).toBeInTheDocument()
```

- [x] Add Playwright responsive check with CDP/browser viewport sizes 390x844, 768x1024, and 1440x900. Assert no `document.documentElement.scrollWidth > clientWidth` on `/moderation/rules`.

Run: `bunx vitest run apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlayground.accessibility.test.tsx`

Run: `npx playwright test apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-responsive.spec.ts --project=chromium`

Expected: all tests pass; 390px viewport has no page-level horizontal overflow.

**Verification:**
- `bunx vitest run ../packages/ui/src/components/Option/ModerationPlayground/__tests__ ../packages/ui/src/services/__tests__/moderation.service.contract.test.ts` - passed, 20 files / 217 tests.
- `bunx playwright test e2e/workflows/tier-5-specialized/moderation-responsive.spec.ts e2e/workflows/tier-5-specialized/moderation-routes.spec.ts --project=tier-5 --reporter=line` - passed, 7 tests.
- `git diff --check` - passed.
- `bunx tsc --noEmit --pretty false` - still blocked by pre-existing unrelated errors in `EmbeddingsModelSelectionConfig.tsx`, `persona-visuals.ts`, and `lib/api/vnPlay.ts`.
- Bandit not run; touched files are TypeScript/TSX, Playwright tests, docs, and Backlog metadata only.

**Status:** Complete

---

## Stage 4: Backend Review Contract, Store, Permissions, And Event Producers

**Goal:** Add durable, sanitized moderation review items and audit events behind explicit review permissions.

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/rbac_seed.py`
- Covered by shared Postgres/SQLite RBAC bootstrap: `tldw_Server_API/app/core/AuthNZ/rbac_seed.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- Create: `tldw_Server_API/app/core/Moderation/review_store.py`
- Create: `tldw_Server_API/app/core/Moderation/review_service.py`
- Covered by adjacent review-safe projection helper: `tldw_Server_API/app/core/Moderation/review_service.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Test: `tldw_Server_API/tests/unit/test_moderation_review_store.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_review_service.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_event_capture.py`

**Implementation Tasks:**

- [x] Add permission constants:

```py
MODERATION_REVIEW_READ = "moderation.review.read"
MODERATION_REVIEW_DECIDE = "moderation.review.decide"
MODERATION_REVIEW_BULK_DECIDE = "moderation.review.bulk_decide"
MODERATION_AUDIT_READ = "moderation.audit.read"
```

- [x] Seed permissions in SQLite and Postgres RBAC paths. Grant all to admin, grant `moderation.review.read` and `moderation.review.decide` to reviewer, and add review permissions to `SINGLE_USER_DEFAULT_PERMISSIONS` so local single-user mode remains usable.

- [x] Refactor `tldw_Server_API/app/api/v1/endpoints/moderation.py` so config endpoints keep admin plus `SYSTEM_CONFIGURE`, but review endpoints use review permissions. Use a root `router = APIRouter()` plus `rules_router` and `review_router` if needed; do not leave review endpoints under a global `SYSTEM_CONFIGURE` dependency.

- [x] Add Pydantic schemas to `moderation_schemas.py`:

```py
ModerationReviewStatus = Literal[
    "needs_review", "approved", "blocked", "redacted", "dismissed", "escalated"
]
ModerationDecisionAction = Literal["approve", "block", "redact", "dismiss", "escalate"]
ModerationSeverity = Literal["low", "medium", "high", "critical"]

class ModerationReviewMatch(BaseModel):
    rule_id: str | None = None
    pattern_type: Literal["literal", "regex", "pii", "category"] | None = None
    category: str | None = None
    action: Literal["pass", "block", "redact", "warn"] | None = None
    sample: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)

class ModerationReviewItem(BaseModel):
    id: str
    status: ModerationReviewStatus
    phase: Literal["input", "output"]
    source_type: str | None = None
    source_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    created_at: str
    updated_at: str | None = None
    severity: ModerationSeverity | None = None
    category: str | None = None
    safe_fields: dict[str, bool] = Field(default_factory=dict)
    excerpt: str
    context: dict[str, str] | None = None
    effective_policy: dict[str, Any] = Field(default_factory=dict)
    matches: list[ModerationReviewMatch] = Field(default_factory=list)
    recommended_action: ModerationDecisionAction | None = None
```

- [x] Create `review_store.py` using SQLite at `MODERATION_REVIEW_DB_PATH` or default `tldw_Server_API/Databases/moderation_review.db`. Use `configure_sqlite_connection`, `sqlite3.Row`, parameterized queries, and schema creation in `__init__`.

- [x] Store only sanitized fields by default. Minimum tables:
  - `moderation_review_items`
  - `moderation_review_decisions`
  - `moderation_review_audit_events`

- [x] Add unique `idempotency_key` on `moderation_review_items` so repeated checks do not create duplicates.

- [x] Add retention fields `retention_expires_at` and `content_redacted_at`. Implement `redact_item_content(item_id, actor_id)` even if only used by tests in this stage.

- [x] Create `review_service.py` with list/detail/record/decision/undo/bulk/audit methods. Map actions to resulting statuses: approve -> approved, block -> blocked, redact -> redacted, dismiss -> dismissed, escalate -> escalated.

- [x] Add review endpoints:
  - `GET /moderation/review/items`
  - `GET /moderation/review/items/{item_id}`
  - `POST /moderation/review/items/{item_id}/decision`
  - `POST /moderation/review/items/{item_id}/undo`
  - `POST /moderation/review/bulk-decision`
  - `GET /moderation/review/audit`

- [x] Add `CurrentPrincipal` dependency to decision endpoints so `decided_by` and audit actor use the authenticated principal id. Never trust actor fields from the request body.

- [x] Extend `ModerationEvaluationResult` or add an adjacent helper in `moderation_service.py` to provide review-safe metadata: sanitized excerpt, matched pattern type/category/action, effective policy snapshot, source phase, and recommended action. Keep raw text out of the returned review payload.

- [x] Add event producer calls in `tldw_Server_API/app/core/Chat/chat_service.py` where input/output moderation already resolves `block`, `redact`, or `warn`. Create review items after the moderation action is known and before returning or raising, using an idempotency key based on source type, source id, phase, user/principal id, category, action, and sanitized sample hash.

- [x] Add a narrow endpoint-level hook in `tldw_Server_API/app/api/v1/endpoints/chat.py` only where moderation is handled outside `chat_service.py`. Avoid duplicate events for paths already routed through `moderate_input_messages`.

- [x] Add `MODERATION_REVIEW_CAPTURE_ENABLED` config/env gate defaulting off for production capture until Stage 5 UI and retention behavior are verified. Tests can enable it explicitly.

- [x] Add review endpoints to `apps/packages/ui/src/services/tldw/openapi-guard.ts`.

**Tests:**

- [x] Store/service tests: schema creation, insert idempotency, sanitized list/detail, filtering, pagination cursor, decision, undo, bulk partial failure, audit order, retention redaction.

- [x] Auth tests: config endpoints still require admin plus `SYSTEM_CONFIGURE`; review list allows `MODERATION_REVIEW_READ`; decision requires `MODERATION_REVIEW_DECIDE`; bulk requires `MODERATION_REVIEW_BULK_DECIDE`; audit requires `MODERATION_AUDIT_READ`.

- [x] Event capture tests: chat input block/redact/warn creates one review item with sanitized excerpt and no raw text; repeated call with same idempotency key does not duplicate.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/unit/test_moderation_review_store.py tldw_Server_API/tests/unit/test_moderation_review_service.py tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py tldw_Server_API/tests/unit/test_moderation_event_capture.py -q`

Expected: tests pass with review capture enabled only in tests that opt into it.

**Verification:**

- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/unit/test_moderation_review_store.py tldw_Server_API/tests/unit/test_moderation_review_service.py tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py tldw_Server_API/tests/unit/test_moderation_event_capture.py -q` -> 21 passed, 5 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py -q` -> 15 passed, 5 warnings.
- `bun run verify:openapi` from `apps/packages/ui` -> 265 ClientPath entries verified, 49 media add fallback fields verified, 10 existing reviewed exception paths allowed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m py_compile tldw_Server_API/app/core/Moderation/review_store.py tldw_Server_API/app/core/Moderation/review_service.py tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py` -> passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/app/core/Moderation/review_store.py tldw_Server_API/app/core/Moderation/review_service.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py -f json -o /tmp/bandit_moderation_review_stage4.json` -> no findings; intentional `nosec B608` suppressions on whitelisted dynamic SQL fragments.
- `git diff --check` -> passed.

**Status:** Complete

---

## Stage 5: Moderation Review Queue MVP

**Goal:** Implement `/moderation` as a usable queue for first-time and returning moderators using the Stage 4 backend contract.

**Files:**
- Modify: `apps/packages/ui/src/services/moderation.ts`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/index.ts`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/ModerationReviewShell.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/hooks/useModerationReviewQueue.ts`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/ReviewQueueToolbar.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/ReviewQueueList.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/ReviewItemDetail.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/DecisionBar.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/ReviewStatePanels.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/review-utils.ts`
- Test: `apps/packages/ui/src/components/Option/ModerationReview/__tests__/ModerationReviewShell.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ModerationReview/__tests__/review-utils.test.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review.spec.ts`

**Implementation Tasks:**

- [x] Add TypeScript types and service functions for all Stage 4 review endpoints in `apps/packages/ui/src/services/moderation.ts`.

- [x] Build `useModerationReviewQueue` with state for `status`, `category`, `severity`, `source_type`, `source_id`, `user_id`, `q`, local `sort`, `cursor`, `selectedItemId`, `loading`, `error`, `partial`, and `warnings`. Date filters are deferred until the backend exposes date query parameters.

- [x] Implement `ReviewQueueToolbar` with status, category/severity, source/user, local sort, search, and refresh. Keep controls dense and label every input.

- [x] Implement `ReviewQueueList` as a responsive table/list hybrid. Desktop uses dense columns; mobile collapses to stacked rows. Required visible fields: status, severity, category, phase, source, user/session, created time, sanitized excerpt, recommended action.

- [x] Implement `ReviewItemDetail` with sanitized excerpt/context, provenance, effective policy, matches, and warnings when safe fields are unavailable. Prior decision history remains Stage 6 audit timeline scope.

- [x] Implement `DecisionBar` with approve, block, redact, dismiss, and escalate. Require reason for block, redact, and escalate; allow optional reason for approve/dismiss. Confirm destructive actions.

- [x] After a successful single-item decision, refresh counts, update selected item, and show an undo affordance when `undo_token` is returned.

- [x] Implement empty, loading, error, permission denied, backend unsupported, and partial-data states in `ReviewStatePanels.tsx`.

- [x] For extension wrapper, render compact queue controls and selected item summary if full table width is unavailable. Include "Open full review" action; do not redirect to rules.

**Tests:**

- [x] Unit tests for service URL/query building and response typing.
- [x] Component tests for empty/loading/error/permission states.
- [x] Component tests for filters updating query state and refresh calls.
- [x] Component tests for decision reason validation and undo display.
- [x] E2E test using mocked review fixture to exercise list -> detail -> decision -> undo.

Run: `bunx vitest run apps/packages/ui/src/components/Option/ModerationReview/__tests__/ModerationReviewShell.test.tsx apps/packages/ui/src/components/Option/ModerationReview/__tests__/review-utils.test.ts apps/packages/ui/src/services/__tests__/moderation.service.contract.test.ts`

Run: `npx playwright test apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review.spec.ts --project=chromium`

Expected: `/moderation` is usable with seeded/mocked review items and all required states render.

**Status:** Not Started

---

## Stage 6: Audit, Recovery, And Content Redaction

**Goal:** Make moderation decisions trustworthy, reversible when allowed, and auditable without exposing unsafe content.

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/review_store.py`
- Modify: `tldw_Server_API/app/core/Moderation/review_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- Modify: `apps/packages/ui/src/services/moderation.ts`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/AuditTimeline.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/ReviewItemDetail.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/DecisionBar.tsx`
- Test: `tldw_Server_API/tests/unit/test_moderation_review_audit.py`
- Test: `apps/packages/ui/src/components/Option/ModerationReview/__tests__/AuditTimeline.test.tsx`

**Implementation Tasks:**

- [ ] Add decision history to item detail response. Include only sanitized actor id, action, resulting status, reason, timestamps, undo eligibility, and redaction state.

- [ ] Ensure undo tokens are stored hashed, expire, and are single-use. Undo should append an audit event and restore the previous status only when no later decision superseded it.

- [ ] Add audit list filters by `item_id`, `decision_id`, `actor`, `action`, `date_from`, `date_to`, `cursor`, and `limit`.

- [ ] Add explicit redaction support for review item content when source data deletion or privacy policy requires it. Redacted items should keep metadata/audit records but replace excerpt/context/match samples with safe placeholders.

- [ ] Build `AuditTimeline.tsx` and surface it in item detail. Keep raw JSON behind a details element only for debugging-safe payloads.

- [ ] Add sanitized audit export service only if the backend endpoint is ready; otherwise leave export as a documented Stage 6 follow-up within the task.

**Tests:**

- [ ] Backend tests for undo expiration, single-use undo, later-decision conflict, audit list filters, and content redaction.
- [ ] Frontend tests for audit timeline, redacted content state, and undo disabled/expired states.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/unit/test_moderation_review_audit.py -q`

Run: `bunx vitest run apps/packages/ui/src/components/Option/ModerationReview/__tests__/AuditTimeline.test.tsx apps/packages/ui/src/components/Option/ModerationReview/__tests__/ModerationReviewShell.test.tsx`

Expected: decisions are auditable and undo/redaction behavior is explicit in UI and backend.

**Status:** Not Started

---

## Stage 7: Power-User Review Efficiency

**Goal:** Support returning moderators who need fast filtering, batching, and keyboard-driven review without sacrificing safety.

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/ModerationReviewShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/ReviewQueueToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/ReviewQueueList.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/DecisionBar.tsx`
- Create: `apps/packages/ui/src/components/Option/ModerationReview/BulkDecisionBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationReview/hooks/useModerationReviewQueue.ts`
- Test: `apps/packages/ui/src/components/Option/ModerationReview/__tests__/BulkDecisionBar.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review-power-user.spec.ts`

**Implementation Tasks:**

- [ ] Add multi-select to queue rows with a visible selected count and clear selection control.

- [ ] Add `BulkDecisionBar` for approve, dismiss, block, redact, and escalate. Require confirmation and reason for destructive/high-risk bulk actions.

- [ ] Wire `POST /moderation/review/bulk-decision` and render partial failure results inline.

- [ ] Add saved filter presets in local storage for status/category/severity/source/date/sort. Keep the persistence local and reversible.

- [ ] Add keyboard shortcuts only when focus is inside the review surface: next/previous item, approve, dismiss, focus search, refresh. Show shortcuts in tooltips or a small help popover, not as permanent instructional copy.

- [ ] Add "review complete" completion state when `needs_review` count reaches zero, with secondary links to audit and content rules.

**Tests:**

- [ ] Unit/component tests for bulk confirmation, required reason validation, partial failure display, saved filter persistence, and keyboard shortcuts.

- [ ] E2E test for selecting several rows, bulk dismissing, handling one failed item, and clearing selection.

Run: `bunx vitest run apps/packages/ui/src/components/Option/ModerationReview/__tests__/BulkDecisionBar.test.tsx apps/packages/ui/src/components/Option/ModerationReview/__tests__/ModerationReviewShell.test.tsx`

Run: `npx playwright test apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review-power-user.spec.ts --project=chromium`

Expected: repeat-review workflow can be completed without opening every item, while destructive actions remain confirmed and audited.

**Status:** Not Started

---

## Stage 8: Fixtures, Regression Coverage, Docs, And Verification

**Goal:** Make the route split and review workflow stable in tests, docs, and future audits.

**Files:**
- Create: `apps/tldw-frontend/e2e/fixtures/moderation-review-items.json`
- Modify: `apps/tldw-frontend/e2e/utils/fixtures.ts`
- Modify: `apps/tldw-frontend/e2e/page-mapping.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/smoke.setup.ts`
- Modify: `Docs/Code_Documentation/Moderation-Guardrails.md`
- Modify: `Docs/API/` moderation docs if an existing generated or hand-authored API doc references moderation endpoints
- Modify: `Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md` only if implementation intentionally diverges from the approved spec
- Modify: Backlog implementation tasks created for this plan

**Implementation Tasks:**

- [ ] Add E2E fixtures for populated review queue, empty queue, permission denied, backend error, partial data, expired undo, and redacted content states.

- [ ] Update smoke/page inventory so canonical routes are `/moderation` and `/moderation/rules`; keep `/moderation-playground` only as a legacy redirect case.

- [ ] Update `Docs/Code_Documentation/Moderation-Guardrails.md` to distinguish "Moderation Review" from "Content Rules", list review permissions, explain sanitized review data, and document retention/minimization behavior.

- [ ] Document `MODERATION_REVIEW_CAPTURE_ENABLED` and `MODERATION_REVIEW_DB_PATH`.

- [ ] Add a short "known unsupported states" note if any review producers remain unwired after the MVP.

- [ ] Run full focused frontend/backend verification and record results in the relevant Backlog tasks.

**Verification Commands:**

- [ ] Backend focused:

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py tldw_Server_API/tests/unit/test_moderation_review_store.py tldw_Server_API/tests/unit/test_moderation_review_service.py tldw_Server_API/tests/unit/test_moderation_review_audit.py tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py tldw_Server_API/tests/unit/test_moderation_event_capture.py -q`

Expected: all focused moderation backend tests pass.

- [ ] Frontend focused:

Run: `bunx vitest run apps/packages/ui/src/components/Option/ModerationPlayground/__tests__ apps/packages/ui/src/components/Option/ModerationReview/__tests__ apps/packages/ui/src/components/Layouts/__tests__/settings-nav.moderation.test.ts apps/packages/ui/src/services/__tests__/moderation.service.contract.test.ts`

Expected: all focused moderation frontend tests pass.

- [ ] Route and responsive E2E:

Run: `npx playwright test apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-routes.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-responsive.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-review-power-user.spec.ts --project=chromium`

Expected: canonical routes, legacy redirect, responsive behavior, review queue, decision, undo, and bulk flows pass through Playwright/CDP.

- [ ] Design-system/state guard:

Run: `bun run verify:design-system-state`

Expected: no new design-system state violations.

- [ ] Security scan on touched backend production files:

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/app/core/Moderation/review_store.py tldw_Server_API/app/core/Moderation/review_service.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py -f json -o /tmp/bandit_moderation_review.json`

Expected: no new findings in touched code.

- [ ] Whitespace/diff hygiene:

Run: `git diff --check`

Expected: no whitespace errors.

**Status:** Not Started

---

## Implementation Risk Checks

- [ ] Review endpoints must not inherit the existing global `SYSTEM_CONFIGURE` dependency, or reviewer role support will be fake.
- [ ] The review UI must treat backend `safe_fields` as authoritative and avoid rendering absent raw fields.
- [ ] Event capture must be idempotent and gated until the UI/retention path is verified.
- [ ] Extension route parity must be explicit; `/moderation` must not silently point to rules.
- [ ] Raw blocklist replace/upload must not bypass lint and preview.
- [ ] Power-user shortcuts must not fire while typing in inputs/textareas.
- [ ] E2E fixtures must cover populated and missing-data states; an empty-only queue is not enough.

## Plan Review Notes

- A separate plan-document-reviewer subagent was not dispatched while writing this artifact because the current collaboration/tool rules only allow spawning subagents when the user explicitly requests delegated agent work.
- Local self-review checklist completed against the approved spec: routes, rules hardening, accessibility/responsive, backend contract, review queue, audit/recovery, power-user workflows, fixtures/docs, and verification are all represented as staged work.
- Before implementation begins, choose subagent-driven execution or inline execution. If subagent-driven execution is chosen, run a fresh review pass on this plan before assigning Stage 1.
