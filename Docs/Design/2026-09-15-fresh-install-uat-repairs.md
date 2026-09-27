# Fresh-install UAT repairs

## Intent and scope

The user approved addressing all findings in the [running tracker](../Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md) before the next full UAT. This design restores the expected behavior recorded there. It changes existing setup, connection, Home, Knowledge QA, and admin flows; it does not introduce a new subsystem. Targeted regression verification is part of the repair work. The next full UAT is deliberately deferred.

## Setup progression (UAT-001–004)

- Treat a provider's selected model ID as an opaque identifier, including path-like local model IDs. Limit any path exception to the providers/default_model field and preserve secret filtering and validation elsewhere.
- First-run readiness requests use the same first-run authorization mode as the other setup calls. Admin readiness continues to require authenticated access.
- First-chat verification uses an explicit inference timeout and presents a useful timeout category. A five-second generic request deadline is inappropriate for a cold local model.
- A multi-user server leads to login settings without a solo-state mutation. Keep CSRF and protected setup endpoints unchanged. A user selecting the multi-user guide from a single-user server can still return to setup paths.

## Account access and administration (UAT-007,008,011,012,014)

- Validate an ordinary multi-user session with the authenticated identity endpoint. Operator health remains protected by system.logs.
- Explicit connection diagnostics use the active authentication mechanism and do not send saved credentials to an edited foreign origin.
- Default to password login, make mode-change confirmation conditional on existing credentials, use clear action labels, and show extension permission controls only inside the extension.
- Add a Create user action to the existing admin Users & roles section, backed by the existing protected create-user API and schema. Default role is user; surface validation/duplicate errors and refresh the list on success.
- Gate billing requests and rendering on the server/deployment capability. A server without billing should not show five separate 404 alerts.
- UAT-016, found after restoring Notes access: request admin note-title settings only for an active administrator, using the existing scoped Notes identity. Ordinary users retain the supported default title behavior.

## First-source and Home (UAT-005,006,010,013)

- Loading provider discovery is distinct from a confirmed empty provider list.
- Successful first-source ingestion updates the shared milestone consumed by Home; previously persisted first-source state is reconciled so reload does not regress the checklist.
- Reuse the existing persisted discuss-media handoff for the source prompt and media context, then navigate to Chat. A DOM event without a mounted consumer is insufficient.
- Preserve optional assistant setup while preventing it from interrupting a completed unified-onboarding source task. Do not treat failed profile loading as proof no profiles exist.
- Auth copy must distinguish backend availability from WebUI runtime exposure. Explain where to find the operator's configured SINGLE_USER_API_KEY when manual recovery is necessary; never expose a disabled credential automatically.
- Capture source/milestone owner scope when the operation starts. Restore progress only for the same account and server; unscoped legacy evidence must not credit another account. Core source destinations may bypass optional personalization without an additional restricted setup probe.

## Knowledge QA and chat settings (UAT-009,015)

- Trace provider aliases and selected provider/model propagation through the QA client and backend. Use existing canonicalization at the appropriate boundary; keep provider egress and credential policy intact.
- Failed searches must clear transient progress and must not become successful prior-answer context. Keep truthful failure history if the existing contract requires it.
- Missing optional chat settings should use a normal default or supported existence contract without a misleading browser error. Do not suppress genuine authorization or service errors.
- Targeted live QA confirmed that keyword search finds the fixture but security filtering excludes it. Preserve the existing classification/ACL policy. Propagate only an aggregate filtered outcome, never excluded document IDs/text, and explain that outcome in the UI instead of calling it no matches or generating an answer with no evidence. Validate successful grounded retrieval separately with a public control fixture. Do not silently remap AuthNZ roles into the standalone RAG ACL or relax sensitivity defaults in this repair.

## Tracking and environment findings

Each product finding receives a code fix or a documented, evidence-backed disposition. Existing environment corrections (isolated Redis/config/PYTHONPATH and host access) remain distinct from product bugs. Use explicit Backlog IDs through the official MCP/CLI to avoid the previous ID-allocation collision; do not repair unrelated historical task IDs in this change.

## Verification and limits

Add behavior-level failing tests before each implementation, run the focused frontend/backend suites, lint the touched scope, and run Bandit for Python changes. Review integration across shared files before committing. Use narrow live checks where needed to verify an observed failure; do not perform the next full workflow UAT yet. Locate the frontend journey/tier definitions and link the actual executable scenarios without inventing A/B/C labels that are absent from source.

## Citation contract follow-up (UAT-017)

The narrow public-source live retest found that streaming generation drops the requested citation option and labels contexts incompatibly with the numeric citation parser. Carry the explicit option into generation and request inline numbered citations against correspondingly numbered sources. Preserve operator prompt text and uncited behavior when disabled; never fabricate citation mappings from prose titles or out-of-range IDs. Verify the actual prompt passed to the provider and then repeat one live source-grounded question.

## Final targeted UX follow-ups (UAT-018–019)

- Preserve raw hybrid ranking scores for ordering, but do not display reciprocal-rank fusion as a calibrated percentage or use it to infer low answer confidence. Explicitly distinguish ranking-only evidence from calibrated relevance and retain valid warnings when a calibrated score exists.
- Opening an ordinary note should not automatically request graph data when `notes.graph.read` is unavailable. Use existing identity/capability mechanisms, preserve backend permissions, and avoid retries for authorization denials. Keep ordinary note editing usable.
