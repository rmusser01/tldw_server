# B02: recover an admitted Character question without duplicating it

Task: TASK-13260.278.18.83.45. Active execution: Docs/Plans/IMPLEMENTATION_PLAN_uat_b02_character_chat_transitions_20261015.md.

## Observed defect

The real unavailable-provider request saved user122995b2-125b-4fe1-bad0-107946bb7d9b in the owned Character conversation, then returned a sanitized provider error before emitting the admission frame. The browser retained an unknown outcome and old three-message view. Its safe recovery preflight found the fourth canonical row and stopped before a duplicate Send. There has been one unavailable-model Send and zero recovery Sends. The old failure and unknown operation remain historical; a later answer must not retrospectively reclassify them.

The new real endpoint/database causal test also fails unchanged: one failed test and actual HTTP502 before an admission frame. No application or runtime repair has yet been qualified by that result. Original tests/assertions and provider sanitization remain.

## Small repair sequence

1. For an already admitted native streaming turn, publish its authentic admission before an allow-listed, sanitized pre-stream provider error. Do not change non-native or pre-admission HTTP failures, perform another provider attempt, manufacture an assistant settlement, or roll back accepted input. Client ownership parsing must retain an accepted-unsent outcome when it actually receives that receipt.
2. Reuse the existing owner-selected input schema and DB settlement for a new, explicitly reviewed answer intent on a canonical unanswered user. Add explicit server-settled completion support and a distinct opt-in capability; the older capture/message-settlement capability does not prove completion support. Refuse unsupported servers before dispatch and never fall back to ordinary Send. Validate owner, full fresh snapshot, cursor after the canonical user, its exact revision/content and frozen Character context, and absence of an existing answer when the UI promises an unanswered question. Skip user admission and settle only the new assistant under the original user ID.
3. For this existing unknown record, first reload/read canonical history normally. Offer review of the actual saved unanswered question and current model/Character context as a new deliberate intent. Keep the old unknown recovery intact; do not reconstruct an admission from a matching string/ID or promote that operation's outcome. A fresh server-validated existing-input selection authorizes the new answer independently.

Use the normal current owner/profile/request lease, cancellation and unsupported Sync-owner restrictions. Current draft text must remain intact. No automatic send, retries or hidden branch/legacy continuation fallback. A changed question, ancestor, scope, Character context, request settings or owner refuses dispatch/settlement rather than selecting a convenient replacement.

## Compatibility and validation

The new completion capability must be explicitly acknowledged. Its request carries an empty current-message list: the canonical selected user is already the only question in model context. Keep the old request schema's nonempty-message rule for every other mode. An older schema therefore rejects this payload even if it ignores an unfamiliar extension field; do not send a duplicate current user as a compatibility substitute. The old request schema ignores extra fields, so merely adding a JSON field is unsafe: an older server must fail closed before any inference or input admission. Test the old-server refusal as well as unsupported owners. Incoming provider bytes cannot supply trusted admission or settlement metadata.

Causal checks: pre-stream provider failure after real admission delivers exactly that receipt and sanitized error/no assistant; non-native behavior remains; fresh existing-question retry calls the provider once, includes the canonical question once, creates no user and exactly one assistant; owner/input/ancestor/context drift refuses; no automatic retry/draft loss; real unavailable-model recovery saves CHIRP under the original question. Use existing endpoint/DB fixtures and the official isolated PostgreSQL fixture for affected persistence checks. Mock the provider transport only in automated tests; final UAT uses the real configured provider.

Touched production/test scope will be recorded as implementation proceeds. Run affected configured UI/WebUI tests/types/lint and Python compile/lint/Bandit baseline/current, canonical schema/type generation for any public contract change, independent source review and necessary changed-source build/runtime qualification. Broad shared-UI semantic, latest-dev/integration/release gates remain open. Do not replay unrelated accepted UAT or consumed generations.

## ADR assessment

No new durable ownership or persistence rule is proposed: reuse ADR-049 owner-selected history, ADR-050 native lifecycle and ADR-059 task editing. The distinct capability enforces the existing fail-closed compatibility rule rather than silently widening an old acknowledgment. Reassess if implementation requires a different authority or persistence rule. Accepted ADR rationale is unchanged.

## First causal result

The new native pre-stream regression passed after the scoped receipt fix:1 passed/20 warnings/7.80 seconds, natural0. The initial red returned502; production now emits the authentic receipt and allow-listed error without a settlement. A distinct legacy HTTP502/no-receipt/sanitization regression was added. The full affected server-settled metadata and authenticated history API suites are running; no runtime API restart, new native retry implementation, or live recovery acceptance yet.
