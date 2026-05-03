# Phase 4.5 API Versioning Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a docs-only Phase 4.5 versioning policy that aligns the existing `v1` contract, shared envelope/pagination helpers, and frontend/client migration rules.

**Architecture:** Keep this tranche policy-only. Update the canonical API docs and roadmap tracker so future route-family work has one explicit decision framework for additive `v1` changes versus sibling-route or `/api/v2/` migrations. Do not change runtime code or endpoint behavior in this plan.

**Tech Stack:** Markdown docs, GitHub issue tracker, focused verification via grep and `git diff --check`

---

## File Map

- Modify: `Docs/API/api-versioning-strategy.md`
- Modify: `Docs/API/Pagination.md`
- Modify: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md`
- Optionally modify: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md`
- Reference only: `tldw_Server_API/app/api/v1/schemas/response_envelope.py`
- Reference only: `tldw_Server_API/app/api/v1/utils/response_envelope.py`
- Reference only: `apps/packages/ui/src/services/response-envelope.ts`
- External update: GitHub issue `#1116`

## Task 1: Update Canonical Versioning Strategy Doc

**Files:**
- Modify: `Docs/API/api-versioning-strategy.md`
- Reference: `Docs/superpowers/specs/2026-05-03-api-versioning-policy-design.md`

- [ ] **Step 1: Write the failing review expectation**

Define the expected additions before editing:

- `v1` remains legacy-default
- additive header opt-in is transitional only
- future default-breaking changes use `/api/v2/`
- provider-compatible and transport exemptions are explicit
- deprecation headers are not implied for additive `v1` pilots

- [ ] **Step 2: Verify the current doc lacks those specifics**

Run:

```bash
rg -n "legacy-default|X-TLDW-Response-Envelope|provider-compatible|metadata.pagination|transitional" Docs/API/api-versioning-strategy.md
```

Expected:

- either no matches or incomplete coverage proving the policy update is needed

- [ ] **Step 3: Write minimal doc updates**

Edit `Docs/API/api-versioning-strategy.md` to add:

- a Phase 3 / Phase 4 compatibility policy section;
- the `v1` legacy-default rule;
- the additive-header-versus-path-version distinction;
- the route exemption classes;
- the deprecation-header rule for approved deprecation windows only.

- [ ] **Step 4: Verify the updated doc contains the new policy anchors**

Run:

```bash
rg -n "legacy-default|X-TLDW-Response-Envelope|/api/v2/|provider-compatible|Deprecation" Docs/API/api-versioning-strategy.md
```

Expected:

- all key policy anchors present

- [ ] **Step 5: Commit**

```bash
git add Docs/API/api-versioning-strategy.md
git commit -m "docs: codify Phase 4.5 API versioning policy"
```

## Task 2: Reconcile Pagination Doc With Shipped Helper Contract

**Files:**
- Modify: `Docs/API/Pagination.md`
- Reference: `tldw_Server_API/app/api/v1/schemas/response_envelope.py`
- Reference: `tldw_Server_API/app/api/v1/utils/response_envelope.py`
- Reference: `apps/packages/ui/src/services/response-envelope.ts`

- [ ] **Step 1: Write the failing review expectation**

Define the expected clarification:

- `pagination` stays the additive nested body field for default `v1` route bodies
- canonical envelope metadata uses `metadata.pagination`
- raw-list and provider-compatible migrations still require sibling-route or
  `/api/v2/` policy decisions

- [ ] **Step 2: Verify the current doc is ambiguous or incomplete**

Run:

```bash
rg -n "metadata.pagination|raw-list|provider-compatible|v2" Docs/API/Pagination.md
```

Expected:

- missing or incomplete references that justify the doc update

- [ ] **Step 3: Write minimal clarification**

Edit `Docs/API/Pagination.md` to:

- explicitly distinguish body-level `pagination` from envelope-level
  `metadata.pagination`;
- reaffirm compatibility rules for raw-list and provider-compatible routes;
- point version-breaking pagination migrations back to the versioning policy.

- [ ] **Step 4: Verify the clarification landed**

Run:

```bash
rg -n "metadata.pagination|raw-list|provider-compatible|versioning" Docs/API/Pagination.md
```

Expected:

- all clarifying anchors present

- [ ] **Step 5: Commit**

```bash
git add Docs/API/Pagination.md
git commit -m "docs: align pagination guide with Phase 4.5 policy"
```

## Task 3: Refresh The Phase 4.5 Decision Packet

**Files:**
- Modify: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md`
- Optionally modify: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md`

- [ ] **Step 1: Write the failing review expectation**

Expected decision-packet fixes:

- replace stale `meta.pagination` language with the shipped `metadata` contract
- include frontend/client boundary rules
- tighten the distinction between additive `v1` behavior and `/api/v2/`
  default-breaking migrations

- [ ] **Step 2: Verify stale wording exists**

Run:

```bash
rg -n "meta.pagination|query opt-in|v2|frontend owner|client" \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md
```

Expected:

- stale or incomplete wording visible

- [ ] **Step 3: Update the review artifacts**

Edit the decision packet, and the alignment doc only if needed, so they match
the new policy spec and no longer conflict with shipped helper terminology.

- [ ] **Step 4: Verify the packet matches the canonical policy**

Run:

```bash
rg -n "metadata.pagination|legacy-default|provider-compatible|/api/v2/" \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md
```

Expected:

- canonical wording present, stale wording removed

- [ ] **Step 5: Commit**

```bash
git add \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md
git commit -m "docs: refresh Phase 4.5 versioning decision packet"
```

## Task 4: Update Roadmap Tracker

**Files:**
- External update: GitHub issue `#1116`

- [ ] **Step 1: Draft the issue comment content**

The comment should state:

- Phase 4.4 is complete and merged
- Phase 4.5 is now active as a policy/docs tranche
- the spec and doc updates define the next decision gate for future response-shape
  migration work

- [ ] **Step 2: Post the tracker update**

Run:

```bash
gh issue comment 1116 --repo rmusser01/tldw_server --body "<finalized update>"
```

Expected:

- GitHub returns the new comment URL

- [ ] **Step 3: Verify the roadmap issue reflects the new status**

Run:

```bash
gh issue view 1116 --repo rmusser01/tldw_server --json comments
```

Expected:

- latest comment reflects Phase 4.5 activation and policy-doc status

## Task 5: Final Verification And Handoff

**Files:**
- Verify all touched docs and tracker references

- [ ] **Step 1: Run focused grep checks**

Run:

```bash
rg -n "legacy-default|metadata.pagination|provider-compatible|/api/v2/|Deprecation" \
  Docs/API/api-versioning-strategy.md \
  Docs/API/Pagination.md \
  Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md
```

Expected:

- all policy anchors present and internally consistent

- [ ] **Step 2: Run whitespace / patch hygiene**

Run:

```bash
git diff --check
```

Expected:

- no output

- [ ] **Step 3: Review branch status**

Run:

```bash
git status --short --branch
```

Expected:

- only intended docs changes, or a clean branch after commits

- [ ] **Step 4: Prepare PR summary**

Summarize:

- canonical `v1` compatibility policy
- `metadata.pagination` reconciliation
- frontend/client boundary rules
- roadmap activation of Phase 4.5
