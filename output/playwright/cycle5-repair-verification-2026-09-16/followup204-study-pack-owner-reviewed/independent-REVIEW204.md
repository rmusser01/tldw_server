# Independent UAT204 / TASK13260.142 review

**CLEAR within the approved one-caller scope. No remaining correctness finding.** Independent official required-PostgreSQL/SQLite run: **30 passed, 0 failed, 0 skipped/deselected, 31.65s**, with four retained existing warnings. No native acceptance, live runtime, model, browser, source, task, or git action was performed.

## Source and causal review

Author manifest `../uat204-repair-20260917/owned-manifest.json` SHA256 `88ac4220398d781dab0bb89cf4390b81697442dc1de15f619bc9206c6a7d0520`; all three source/test hashes match before and after the independent run. Production worker SHA256 `8723ee86e578b05c45bb9a57ce2fd04feaebd6744dc5fc006e144ea0c0cdc78c`.

Independent AST comparison of the exact retained production baseline proves the only production change removes the optional `client_id=study-pack-worker-{id}` keyword from the existing `get_chacha_db_for_user_id` call. The helper already defaults to `str(user_id)`, retains runtime caching and default-character maintenance, and shares the same canonical identity used by the normal owner accessor. Input conversion, independent operation scope, source resolution, persistence, error propagation, and cleanup remain unchanged. The existing media-error test double adapts to the new call signature while preserving the cleanup/error assertions.

The author's retained real-factory RED has two cold failures and two warm passes: cold PostgreSQL persists the custom owner prefix, canonical owner2 cannot read it, the later owner loader reuses that object, and the restricted-role source reader cannot resolve owner2's note. Cold SQLite demonstrates the same cache-dependent new-row label while remaining readable through its per-file boundary. This is meaningful causal evidence, not a mock assertion of the deleted keyword. I inspected that receipt; I did not independently replay RED.

## Independent verification

`uat204-independent-sidebar-command.json` records the exact official runner command. It includes the new four-case cold/warm PostgreSQL/SQLite real factory suite, unchanged worker operation-lifecycle suite, and existing StudyPack worker tests. The new tests execute actual dependency caching, actual source resolver and service persistence through the worker; only external model output, provider selection, Media DB seam, and isolated paths/runtime globals are supplied by the fixture. The fake model must receive the real note evidence and yields one cited card.

The PostgreSQL restricted-role control verifies NOSUPERUSER/NOBYPASSRLS, actual existing note RLS, source resolution, and canonical deck/card writes on the selected worker DB. It is a focused post-acquisition role control, not a complete cold login/bootstrap/model job under that restricted role. SQLite checks that historical rows bearing the old prefix stay readable and retain their label.

Independent static results: all three files compile; Bandit production and both tests have zero findings/errors (B101 excluded for tests only). Ruff has one unchanged I001 import-order diagnostic in the preexisting worker test, identical to its baseline; production and new test are clean. `ast-equivalence.json`, Ruff/Bandit JSON, source-before/after manifests, command and redacted log are retained here.

## Boundaries and limitations

The exact surrounding ChaCha source during this run was the frozen UAT198 candidate `6089c5e0cac45fd0dd2c5253ad906b20108746e35a9189bf3934fc1eca9506b7`; dependency source was `deab1710097b472b595ccad93f22862e30ffb299a5df39eacae0f9e5cba19bef`; unchanged lifecycle test was `554d98227c8884a8ba326f957a021c60df93e196d2c24736bc16d513df1c0ba4`. These are explicit tested boundaries, not a clean whole-HEAD claim.

The repair prevents this caller from creating a new wrongly owned cold cache entry. It does not rewrite existing PostgreSQL prefix-owned rows or repair an already published wrong cache object, nor does it change other callers' optional-client identity contract. Those limits match the approved design. No generic cache change or Flashcards owner-predicate weakening is required. Parent owns integration and native acceptance.
