# UAT226 — TASK13260.165

## Problem and repair

Published pending suggestions contain reconstructed `SuggestionEvidenceExcerpt` dataclass instances. The route passed those directly to the strict nested `SuggestionEvidenceResponse` model, which rejects dataclasses by default. Real SQLite and PostgreSQL requests therefore returned 500 whenever nonempty evidence survived filtering. Empty and rejected evidence concealed the problem.

The only production change is in `list_suggestions` in the existing endpoint: explicitly project the six known evidence fields (`side`, `note_id`, `field`, `start_offset`, `end_offset`, `text`) to dictionaries, matching the surrounding explicit suggestion-item projection. No schema, request model, extra-field policy, bounds, owner selection, fingerprint filter or generic serializer changes.

## Evidence

- Original Stage A expanded pending-list capture: actual PG/SQLite 2 FAIL; private TestClient exception instrumentation proves Pydantic `model_type` at `items.0.evidence.0` / `.1`. Original files and test snapshot retained.
- Dedicated initial24 run:14 FAIL/10 PASS. Four ownership setup attempts violated the existing composite foreign keys when changing a parent note owner. These are fixture errors, not claimed product failures. The corrected owner control uses a real second-owner DB instance and actual authenticated route; offset filtering uses a valid evidence offset outside the note content.
- Clean unchanged endpoint RED:12 FAIL/12 PASS,0 skips,17.72s. All twelve failures are actual HTTP500 on surviving evidence. Positive controls cover foreign source404, changed-source empty result, evidence field/range/text/extra rejection and unchanged strict input schema.
- First mixed149 run exposed existing transport fixture global FakeAPI pollution (38 FAIL/111 PASS). Stage A's owned fresh DB fixture now explicitly selects the real factory through scoped monkeypatch; this does not change production or the existing endpoint test. Final combined backend: **149 PASS / 0 skips**, 85.22s, including all24 new UAT226 controls and the Stage A plus adjacent API tests.

The permanent24 cases cover registered and unbound scopes on both real backends, exact reconstructed serialized excerpts, deleted/stale/oversized/out-of-range target evidence suppression, foreign-owner and changed-source controls, plus closed bounded response/input schemas. No provider transport is called.

## Static/source review

AST comparison identifies only `list_suggestions` changed in the endpoint. Ruff0; Bandit production0 findings/0 errors, test0 with B101 excluded only; both Python files parse/compile. Exact endpoint baseline/snapshot, test snapshot, owned.patch and separate manifest are in this packet. Stage A source is separately attributed; UAT225 remains open for full inactive-Sync lifecycle work.

Use the combined official required-PG/SQLite command in the Stage A report or the retained `.tmp/fresh-uat-recovery-20260916/uat225-226-isolated-final-green-command.json`, with a new evidence label. No native/UI/runtime/DB provisioning/config/git/task actions were performed.
