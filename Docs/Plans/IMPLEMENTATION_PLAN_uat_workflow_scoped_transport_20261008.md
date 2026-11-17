# Workflow scoped transport repair

Associated task: TASK-13260.278.18.83.45 (root-owned; C12 transport unit recorded before edits).

**Goal:** Permit represented canonical Workflow editor requests through the existing captured-scope transport without removing their ownership checks.
**Source contract:** `services/tldw/workflows.ts`, `services/background-proxy.ts`, and the registered routes in `tldw_Server_API/app/api/v1/endpoints/workflows.py`.
**Architecture:** Extend only the shared `isServicePromptRequestPath` path/method allowlist. Retain canonical-path parsing, target/principal/single-user key/refresh-lineage checks, expected-user headers, cancellation, and extension scope payloads. No unscoped fallback, configuration swap, blanket prefix, new transport, or backend change.
**ADR assessment:** ADR required: no. The [ADR index](../ADR/README.md) was searched. This restores existing captured request ownership for represented canonical API operations and creates no new durable architecture rule. Unrepresented catalog/admin/auth/ad-hoc/retry/artifact routes remain outside this scoped allowance.

## Stage 1: Reproduce the canonical service/boundary failure
**Goal:** Add real workflow-service through background-proxy/request-core boundary tests, using only synthetic storage/runtime/fetch adapters.
**Success Criteria:** Canonical Create/Version fail for the actual pre-HTTP Service Prompt path refusal before production edits; malformed and unrepresented route refusal controls pass. Cover every represented scoped client operation, credential drift, Quickstart cookie transport, and extension scope forwarding.
**Tests:** Save/create/version; list/get/run/status/steps; pause/resume/cancel; approve/reject; investigation/attempts/preflight; expected-user/body/signal preservation; same-owner current credentials; target/account/key/lineage drift; forbidden methods/admin/unknown/dot/slash/backslash/absolute paths and encoded unsafe client IDs.
**Status:** Complete

Before production edits, 44 malformed/unrepresented/unsafe-identifier refusal controls passed. The initial filter missed the quoted object-case titles and did not establish a red. Corrected Create/Version selection then produced two exact causal failures through the actual workflow service and background-proxy at its pre-dispatch path guard (natural exit 1, 1.036 seconds; no setup failure).

## Stage 2: Add exact canonical pairs
**Goal:** Change only `service-prompt-scope-error.ts`.
**Success Criteria:** Positive numeric definition IDs and canonical single-segment run/step IDs reach the existing transport; other Workflow paths/methods still refuse. No ownership function or pathname parser changes.
**Tests:** Causal tests green, including actual direct fetch and extension payload authority; all existing security fences retained.
**Status:** Complete

The sole production change adds sixteen allowlist lines. Numeric definition IDs must be positive; run/step segments retain the existing canonical parser's rejection of encoded slash/dot/backslash and malformed paths. The extension worker also imports this same shared predicate; its scope-resolution and authorization functions are unchanged.

## Stage 3: Qualify and freeze
**Goal:** Run affected normal configured UI/Web tests and baseline/current scoped ESLint, then freeze exact production/test hashes for independent review.
**Success Criteria:** Existing installed Node 20.19.5 / Vitest 4.0.18; activated Python 3.12 wrapper; 600-second outer / 300-second case limits; no new scoped lint/parse findings. Maximum three distinct attempts. Keep prior C06 source/tests frozen. Root owns combined types/build/live validation and final independent review.
**Tests:** New actual boundary module plus connected existing scope policy, Web Clipper boundary, and Workflow service contracts. TypeScript-only Bandit N/A. No installation/runtime/browser/API/model/private-data/task/Git actions.
**Status:** In Progress

Current Node qualification: new module 84/84 UI green (natural exit 0, 0.764 seconds); all four affected files 287/287 Web green (natural exit 0, 1.291 seconds). The preceding full UI attempt passed 286 cases and failed only a new cookie fixture's absolute-versus-relative URL expectation; source inspection confirmed the correct same-origin relative fetch contract. The expectation now compares the resolved URL against the page origin, with credentials/CSRF/expected-user/no-key assertions retained. Its 203 unchanged existing UI cases were not replayed. No original tests/assertions were changed. Pre/post scoped ESLint both returned zero errors/fatal errors/warnings, with the unchanged apps-base Next pages-directory advisory. Bandit N/A for TypeScript.

The wrapper mismatch was resolved without installation or environment changes: the existing `.venv-uat-fixall-py312-20261007` was located and verified as Python 3.12.13, then activated for one final normal configured UI/Web affected pair. Both passed all 287 cases across four files, natural exit 0, 2.028 / 1.357 seconds, with no skips. Node 20.19.5 / Vitest 4.0.18; Python only supplied the 600-second subprocess outer guard and reporting. The 300-second case cap was supplied explicitly. Earlier Python 3.11 wrappers are not counted as satisfying that requested execution contract. Independent review and root-owned combined types/build/live validation remain pending.

Frozen production/test SHA256:
- shared guard: `a8dd45fc406068d2d32bf876242d61668ceab51619980da3da577436f7ef876f`
- new boundary module: `3d9a2cac394d448be7ff9e75c132b9422b841f47ee129034a822596add45ced9`

Coverage limits: synthetic storage/runtime/fetch adapters exercise actual workflow services, background-proxy and direct request-core. Extension cases prove request payload scope forwarding; they do not execute a real extension worker or backend. There is no live Save/run/approval/persistence acceptance or all43 completion claim. The prior C06 production and both test hashes remain unchanged.
