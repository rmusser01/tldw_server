# OpenAlex Credential Evidence Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close TASK-12968.7 by proving that OpenAlex remains API-key gated, recording dated official evidence, and preserving an exact reproducible inventory freeze.

**Architecture:** Keep the existing `credentialed_out_of_scope` classification and `api_key` route unless official evidence disproves them. Strengthen only the OpenAlex ledger row, its semantic regression, the derived freeze report, and TASK-12968.2's binding no-secret/no-dispatch handoff; do not add credential loading or runtime OpenAlex execution.

**Tech Stack:** Frozen JSON inventory artifacts, the existing Node semantic validator, JSON Schema validation through the project Python virtual environment, Node's built-in test runner, pytest, and Backlog.md CLI.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/research-source-catalog-deep-research`; never edit the dirty root worktree.
- Treat `https://developers.openalex.org/api-reference/authentication`, `https://developers.openalex.org/api-reference/introduction`, and the dated 2026-02-24 OpenAlex pricing announcement as the official evidence set.
- Preserve `resolution=credentialed_out_of_scope`, `resolution_code=credential_required_no_public_route`, and `credential_requirement=api_key`; anonymous demo capacity is not a production-capable credentialless route.
- State explicitly that a free API key is still a credential under `credentialless-public-v2`.
- Preserve the reconciled totals of 191 mapped and 35 credentialed rows.
- Change hashes and the freeze report only when the ledger artifact actually changes; regenerate with `--as-of 2026-07-13` and trusted reviewer `codex-task-12968.1-source-triage`.
- TASK-12968.2 must keep OpenAlex V2 typed unavailable/skipped and produce zero executable attempts, physical reservations, or gateway calls. It must not add a secret-reference field or positive credentialed branch; authenticated enablement is deferred to a separately authorized future program. V1 behavior remains unchanged. This task adds no secret value, secret loader, credential header, or authenticated request.

---

### Task 1: Harden and freeze OpenAlex credential evidence

**Files:**
- Create: `Docs/superpowers/plans/2026-07-14-openalex-credential-evidence-hardening-implementation-plan.md`
- Modify: `Helper_Scripts/tests/validate_research_source_inventory.test.mjs`
- Modify: `Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json`
- Modify: `Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json`
- Modify: `Docs/superpowers/plans/2026-07-13-research-discovery-foundation-implementation-plan.md`
- Modify through Backlog.md CLI: `backlog/tasks/task-12968.7 - Verify-and-harden-OpenAlex-credential-evidence-in-frozen-research-inventory.md`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: `validateInventoryDocuments(...)`, `canonicalJson(...)`, and `sha256(...)` from `Helper_Scripts/validate_research_source_inventory.mjs`.
- Produces: a ledger row whose official evidence and policy reasoning are exact-regeneration tested, plus a binding TASK-12968.2 requirement that OpenAlex V2 selection remains typed unavailable/skipped with no executable attempt, physical reservation, or gateway call.

#### Stage 1: RED evidence contract

**Goal:** Make the missing dated evidence and free-key policy distinction fail deterministically.

**Success Criteria:** The authoritative inventory test requires all three official references, the 2026-02-24 date, no credentialless OpenAlex route, and language distinguishing a free credential from credentialless access.

**Tests:** Node semantic/exact-regeneration test.

**Status:** Complete

- [x] Extend the authoritative test with these assertions before changing the ledger:

```js
assert.equal(openAlex.resolution, "credentialed_out_of_scope");
assert.equal(openAlex.resolution_code, "credential_required_no_public_route");
assert.ok(openAlex.route_candidates.every((route) => route.credential_requirement !== "none"));
assert.equal(openAlex.route_candidates[0].credential_requirement, "api_key");
assert.match(openAlex.resolution_reason, /free API key is still a credential/i);
const officialEvidence = new Map(
  openAlex.evidence
    .filter((entry) => entry.kind === "resolution_review")
    .map((entry) => [entry.reference, entry.claim]),
);
assert.match(
  officialEvidence.get("https://blog.openalex.org/openalex-api-new-features-and-usage-based-pricing/"),
  /2026-02-24/,
);
assert.match(
  officialEvidence.get("https://developers.openalex.org/api-reference/authentication"),
  /anonymous (?:trial|demo) budget/i,
);
assert.match(
  officialEvidence.get("https://blog.openalex.org/openalex-api-new-features-and-usage-based-pricing/"),
  /no-key calls.*demo.*unsuitable for production/i,
);
assert.match(
  officialEvidence.get("https://developers.openalex.org/api-reference/introduction"),
  /api_key.*required/i,
);
```

- [x] Run `node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs` and record the expected failure on missing dated evidence/policy wording.

**RED evidence (2026-07-14):** The authoritative suite reported 17 passed / 1 failed. The OpenAlex assertion failed because `resolution_reason` did not match `/free API key is still a credential/i`; the ledger still lacked the required explicit policy distinction and the newly asserted dated/reference evidence.

#### Stage 2: Minimal ledger and handoff update

**Goal:** Add only the official evidence and the future executor's fail-closed requirement.

**Success Criteria:** The OpenAlex row has the same classification/count contribution, three official evidence references, dated pricing evidence, and a policy reason that does not conflate zero-cost credentials with credentialless access. TASK-12968.2 keeps OpenAlex V2 typed unavailable/skipped with no attempt, reservation, or gateway call and adds no secret-reference interface or positive credentialed branch.

**Tests:** Re-run the Stage 1 Node test after updating derived hashes/report in Stage 3.

**Status:** Complete

- [x] Update the OpenAlex resolution reason, route coverage notes, credentialless-review notes, and resolution-review evidence claims with this policy meaning: the 2026-02-24 announcement requires a free key for production use, anonymous no-key calls are demo-only, and a free API key remains a credential under `credentialless-public-v2`.
- [x] Add one `resolution_review` evidence entry for the API overview and one for the dated pricing announcement; retain the authentication reference.
- [x] In TASK-12968.2's registry/planner stage, require OpenAlex V2 to carry no secret material or secret-reference interface, remain typed unavailable/skipped, and yield zero attempts, reservations, and gateway calls. Add the exact future regression case to its test-first steps and defer all positive credentialed behavior.

#### Stage 3: Exact regeneration and compatibility

**Goal:** Recompute only values derived from the changed ledger and prove surrounding frozen behavior is intact.

**Success Criteria:** `rows_sha256`, the ledger digest, and the checked-in report exactly match validator output; counts remain 191/35; schema and legacy selection tests pass.

**Tests:** Node semantic/exact report, Python schema, legacy selection.

**Status:** Complete

- [x] Recompute `rows_sha256` with the existing `canonicalJson` and `sha256` functions, then update only the ledger's derived row digest.
- [x] Run the authoritative validator with:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . \
  --gate contract \
  --as-of 2026-07-13 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --json
```

- [x] Update the checked-in freeze report only for changed derived digests, then run:

```bash
node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q Helper_Scripts/tests/test_research_source_inventory_schema.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_selection_contract.py
```

- [x] Confirm validator output reports `mapped=191`, `credentialed_out_of_scope=35`, no errors, and exact equality with the checked-in report.

**GREEN evidence (2026-07-14):** The authoritative Node suite passed 18/18, including exact equality with the checked-in report and substantive assertions for all three official OpenAlex claims. The focused Python schema and legacy-selection suites passed 22/22. Validator output had no errors, reported 191 mapped / 35 credentialed rows, and produced ledger digest `9ed48e61e14298c079fc7371b1c709d027bfefa716a06084869635dc8ce63c10` with rows digest `cd48707690c211d4c2517e03fe21261994f21de59a90dc081d806a2b9efb8cd1`.

#### Stage 4: Review and finalization

**Goal:** Close the prerequisite with reproducible evidence and no hidden runtime expansion.

**Success Criteria:** Diff checks pass, independent task review approves spec and quality, Backlog acceptance/DoD are checked, and the progress ledger records the commit range and review result.

**Tests:** `git diff --check`; Bandit documented as not applicable because no production Python code is touched.

**Status:** Complete

- [x] Run `git diff --check` and review the complete diff for unrelated changes or count drift.
- [x] Request independent spec and quality review; resolve every Critical or Important finding and re-review.
- [x] Mark all four stages Complete, append TASK-12968.7 verification/review evidence to `.superpowers/sdd/progress.md`, and finalize the task through Backlog.md CLI.
- [x] Commit the reviewable unit with `docs(research): harden OpenAlex credential evidence`.

**Review/finalization evidence (2026-07-14):** Independent task review found the implementation spec compliant and approved its quality with no Critical, Important, or Minor findings. The controller independently reran the Node inventory suite (18/18), schema plus legacy-selection pytest suite (22/22), and authoritative validator (`errors=[]`, 191 mapped / 35 credentialed, exact checked-in report). TASK-12968.7 is Done with all acceptance and DoD items checked; duplicate CLI final-summary end markers were removed under the approved narrow tracking repair.

**Local review evidence (2026-07-14):** `git diff --check` passed. The complete diff changes only OpenAlex evidence/reasoning, derived digests, exact semantic assertions, the V2-only fail-closed handoff, and controller-created task/plan tracking. Counts remain 191 mapped / 35 credentialed; no runtime, credential-loading, secret, schema, or production Python code changed. Bandit is not applicable. Independent review and Backlog finalization completed as recorded above.
