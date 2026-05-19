# Phase 3.2 Pagination Medium Tranche Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared pagination helpers and migrate the `skills` plus `slides` offset pilots to expose canonical nested pagination metadata without breaking current request or response shapes.

**Architecture:** Introduce a shared offset-pagination schema/helper layer first, then adopt it in two low-risk route families. Preserve existing query params and top-level pagination fields during the compatibility window while teaching only the touched frontend callers to accept canonical nested `pagination`.

**Tech Stack:** FastAPI `Query`, Pydantic, pytest, Bandit, TypeScript UI service/client code, Vitest

---

### Task 1: Shared Pagination Contract

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/pagination.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`
- Test: `tldw_Server_API/tests/Utils/test_pagination_contract.py`

- [ ] **Step 1: Write failing helper tests**

Add tests for:
- canonical `limit`/`offset`
- `page` + `per_page`
- `page` + `results_per_page`
- canonical values winning over aliases
- `has_more` and `next_offset`
- RFC5988 `Link` headers generated from normalized offset inputs

- [ ] **Step 2: Run focused helper tests to verify red**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
```

- [ ] **Step 3: Add minimal shared schemas and helpers**

Implement:
- `OffsetPaginationMeta`
- internal normalized request structure for offset pagination
- normalization helper that accepts `limit`, `offset`, `page`, `per_page`, `results_per_page`
- metadata builder
- Link header builder that reuses normalized values

- [ ] **Step 4: Run helper tests to verify green**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
```

- [ ] **Step 5: Commit shared pagination helper layer**

```bash
git add tldw_Server_API/app/api/v1/schemas/pagination.py tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py tldw_Server_API/tests/Utils/test_pagination_contract.py
git commit -m "Phase 3.2: add shared offset pagination helpers"
```

### Task 2: Skills Backend Pilot

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
- Test: `tldw_Server_API/tests/Skills/integration/test_skills_api.py`

- [ ] **Step 1: Add failing skills pagination tests**

Cover:
- canonical nested `pagination`
- legacy top-level `count`, `total`, `limit`, `offset` still present
- legacy requests still work
- `page`/`per_page` compatibility alias accepted if adopted in this tranche

- [ ] **Step 2: Run focused skills tests to verify red**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "list_skills and pagination" -q
```

- [ ] **Step 3: Adopt shared pagination helper in skills list**

Keep:
- existing `limit` / `offset`
- existing top-level fields

Add:
- canonical nested `pagination`

- [ ] **Step 4: Run focused skills tests to verify green**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "list_skills and pagination" -q
```

- [ ] **Step 5: Commit the skills backend pilot**

```bash
git add tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/tests/Skills/integration/test_skills_api.py
git commit -m "Phase 3.2: add canonical skills pagination metadata"
```

### Task 3: Slides Backend Pilot

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/slides_schemas.py`
- Test: `tldw_Server_API/tests/Slides/test_slides_api.py`

- [ ] **Step 1: Add failing slides pagination tests**

Cover:
- `/slides/presentations`
- `/slides/presentations/search`
- `/slides/styles`
- `/slides/presentations/{presentation_id}/versions`

Assertions:
- canonical nested `pagination`
- existing top-level `total` or `total_count`, `limit`, `offset` still present

- [ ] **Step 2: Run focused slides tests to verify red**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Slides/test_slides_api.py -k "pagination or styles_list" -q
```

- [ ] **Step 3: Adopt shared pagination helper in slides offset endpoints**

Keep each route’s current top-level fields for compatibility.

- [ ] **Step 4: Run focused slides tests to verify green**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Slides/test_slides_api.py -k "pagination or styles_list" -q
```

- [ ] **Step 5: Commit the slides backend pilot**

```bash
git add tldw_Server_API/app/api/v1/endpoints/slides.py tldw_Server_API/app/api/v1/schemas/slides_schemas.py tldw_Server_API/tests/Slides/test_slides_api.py
git commit -m "Phase 3.2: add canonical slides pagination metadata"
```

### Task 4: Frontend Compatibility Updates

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/*` if direct parser tests live there
- Add or modify: focused Vitest tests for `listVisualStyles()` and skills list parsing

- [ ] **Step 1: Add failing frontend compatibility tests**

Cover:
- legacy top-level skills pagination shape
- canonical nested `pagination` skills shape
- legacy slides `total_count`
- canonical slides `pagination.total`

- [ ] **Step 2: Run focused Vitest to verify red**

Run:
```bash
cd apps/packages/ui && bunx vitest run <focused-skills-and-slides-tests>
```

- [ ] **Step 3: Update parsers without broad client churn**

Only touched callers should learn:
- `pagination.total`
- `pagination.limit`
- `pagination.offset`
- `pagination.has_more`
- `pagination.next_offset`

- [ ] **Step 4: Run focused Vitest to verify green**

Run:
```bash
cd apps/packages/ui && bunx vitest run <focused-skills-and-slides-tests>
```

- [ ] **Step 5: Commit frontend compatibility updates**

```bash
git add apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/components/Option/Skills apps/packages/ui/src/components/Option/PresentationStudio
git commit -m "Phase 3.2: teach skills and slides clients canonical pagination"
```

### Task 5: Final Verification And PR Prep

**Files:**
- Verify touched files only

- [ ] **Step 1: Run backend verification**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Utils/test_pagination_contract.py \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Slides/test_slides_api.py -q
```

- [ ] **Step 2: Run frontend verification**

```bash
cd apps/packages/ui && bunx vitest run <focused-skills-and-slides-tests>
```

- [ ] **Step 3: Run Bandit on touched Python files**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/schemas/pagination.py \
  tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py \
  tldw_Server_API/app/api/v1/endpoints/skills.py \
  tldw_Server_API/app/api/v1/endpoints/slides.py \
  -f json -o /tmp/bandit_phase3_2_medium_tranche.json
```

- [ ] **Step 4: Run git hygiene**

```bash
git diff --check
git status --short --branch
```

- [ ] **Step 5: Push the branch and refresh tracker state**

```bash
git push --force-with-lease origin worktree-phase3.2-pagination-standardization
```
