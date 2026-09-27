# Stage 4 — Test trees, CI reachability, and the `apps/mcp-unified` extraction boundary

## Scope

The specific obligation for this module: `pyproject.toml` `testpaths` includes
`tldw_Server_API/app/core/MCP_unified/tests`, which no other module does. Is that split a
coherent boundary or an accident? Answered by finding what the in-app tests actually test,
and by checking which of them any CI gate runs. Plus the module's single C1 site.

## Code Paths Reviewed

- `pyproject.toml:638` — `testpaths = ["tldw_Server_API/tests", "tldw_Server_API/app/core/MCP_unified/tests"]`.
- `pyproject.toml:642-663` — the `plugins` key with its own comment noting pytest silently
  ignores it; plugins actually load via `pytest_plugins` in the two conftests. Not a finding
  (it is documented in place), but it is why the in-app tree's `conftest.py` is load-bearing.
- `apps/mcp-unified/` — `pyproject.toml`, `LICENSE`, `README.md`, `USER_GUIDE.md`,
  `pytest-artifact-gate.ini`, `src/mcp_unified/` (125 files, 46,521 LOC). **No `tests/`.**
- `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py:1-60` — the
  packaging ratchet: `STANDALONE_PROJECT_ROOT = REPO_ROOT / "apps" / "mcp-unified"` (`:36`),
  builds sdists/wheels via `mcp_unified_artifact_test_utils.build_standalone_distributions`
  (`:38-40`), inserts `apps/mcp-unified/src` on `sys.path` (`:53-54`), then
  `import mcp_unified` (`:56`). Marked `pytestmark = pytest.mark.unit` (`:29`) while it
  drives `subprocess`, `tarfile`, `zipfile` and `importlib.metadata`.
- `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py:1-70` — the AST
  contract ratchet over both trees: `MCP_ROOT` (`:15`) and
  `STANDALONE_MCP_ROOT = REPO_ROOT / "apps" / "mcp-unified" / "src" / "mcp_unified"` (`:17`),
  with named expectations (`EXPECTED_INTERFACE_FILES :19`, `EXPECTED_FAILURE_SYMBOLS :20`,
  `EXPECTED_FAILURE_REASON_CODES :21-27`, `SAFE_EXCEPTION_LOG_HELPERS :43-47`,
  `RUNTIME_POLICY_FIELD_PATHS :55-66`).
- `tldw_Server_API/app/core/MCP_unified/tests/conftest.py:12-38` — in-app fixtures
  (`mcp_ws_client`, `ws_client`), distinct from `tldw_Server_API/tests/MCP_unified/conftest.py`.
- `.github/workflows/ci.yml:1758-1760` (and the identical shard definitions at `:3260`,
  `:4630`, `:5923`, `:7222`) — the `platform-mcp-core` shard:
  `paths: tldw_Server_API/tests/MCP` + `tldw_Server_API/tests/MCP_unified`.
- `.github/workflows/backend-required.yml:185-195` — the required backend gate runs
  `pytest -m "unit and not e2e and not jobs" tldw_Server_API/tests/unit`.
- `.github/workflows/coverage-required.yml:153-157` — the required coverage gate runs
  `tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests`.
- `.github/workflows/mcp-unified-rc.yml:1-35` — triggers on `apps/mcp-unified/**` plus
  **11 named** in-app test files and one fixtures glob; additionally admission-gated on
  `vars.LICENSE_FIRST_CI_ENABLED == 'true'` (`:43-45`) and a `workflow_run` of
  *Frontend License Gate Audit* (`:30-32`).
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py:
  encode_prompt_cursor (66-91)` and `decode_prompt_cursor (94-160)` — the module's single
  C1 site: `base64.urlsafe_b64encode(...).rstrip("=")` at `:91`, repadded with
  `"=" * (-len(raw_cursor) % 4)` at `:115`, over
  `json.dumps(payload, sort_keys=True, separators=(",", ":"))` at `:89`, with a `_CURSOR_VERSION`
  check at `:120-123` (`_CURSOR_VERSION = 1` at `:41`).
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_receipts.py:83, 91`
  — the neighbouring base64 usage, which keeps padding and therefore needs no repad idiom.

## Tests Reviewed

This stage's subject **is** the tests, so the review is the inventory itself; the full
classification is in `2026-09-21-stage1-test-inventory.txt`.

- **Tree A** — `app/core/MCP_unified/tests/`, 147 files / 90,042 LOC. Of these, 35 files /
  39,533 LOC import the standalone `mcp_unified` package and do not test this module at all;
  105 import `core.MCP_unified`; 6 import both; 13 import neither (helpers, packaging and
  policy-store contracts).
- **Tree B** — `tldw_Server_API/tests/MCP_unified/`, 64 files, plus `tldw_Server_API/tests/MCP/`
  (4 files). Tree B's contents are thematically distinct — `test_mcp_hub_*` (28 files),
  `test_*_sanitization.py` (9 files), governance packs, tool catalogs — i.e. hub, governance
  and HTTP-surface tests, whereas Tree A holds module-level and gateway-package tests. That
  split is *defensible on content*; it is undocumented and unenforced.
- The two `test_run_command_module.py` files (one per tree) collect cleanly together — no
  pytest basename collision (stage 1, Validation Commands).
- `test_runtime_package_boundary.py` and `test_extraction_contracts.py` are the two highest-churn
  files in the whole module (50 and 43 commits/12 mo). They are real, well-built ratchets
  guarding an intentional extraction, consistent with ADR-033's "must pass installed-artifact
  and downstream-consumer tests". **They downgrade the risk on the extraction itself** —
  which is why finding 12 is about placement and packaging, not about the extraction being
  unsound.
- One test in Tree A is **red on HEAD**: `test_filesystem_module.py::test_filesystem_glob_marks_file_size_unavailable`
  (stage 2, finding `mcp-unified-5`). It has been red since it was added on 2026-06-03.

## Validation Commands

```
$ grep -n "testpaths" pyproject.toml
638:testpaths = ["tldw_Server_API/tests", "tldw_Server_API/app/core/MCP_unified/tests"]

$ grep -c "tldw_Server_API/app/core/MCP_unified/tests" .github/workflows/ci.yml
0

$ sed -n '1757,1760p' .github/workflows/ci.yml
          - name: platform-mcp-core
            paths: >-
              tldw_Server_API/tests/MCP
              tldw_Server_API/tests/MCP_unified

$ grep -rho "tldw_Server_API/app/core/MCP_unified/tests/[A-Za-z0-9_.*/]*" .github/workflows/ | sort -u | wc -l
      12
  (11 named .py files + the fixtures/mcp_protocol/** glob, all inside mcp-unified-rc.yml,
   and all as *path triggers* rather than as a test selection)

$ sed -n '193,195p' .github/workflows/backend-required.yml
          pytest -q --disable-warnings -p pytest_asyncio.plugin \
            -m "unit and not e2e and not jobs" \
            tldw_Server_API/tests/unit

$ sed -n '154,157p' .github/workflows/coverage-required.yml
          pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
            -m "not jobs and not e2e" \
            tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
            --cov=tldw_Server_API/app --cov-report=xml --cov-report=term-missing --cov-fail-under=12

$ ls apps/mcp-unified
LICENSE  pyproject.toml  pytest-artifact-gate.ini  README.md  src  USER_GUIDE.md
$ find apps/mcp-unified -type d -name tests -not -path '*__pycache__*'
(no output)

$ python -m pytest -q -p no:randomly tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
================== 1 failed, 103 passed, 19 warnings in 8.28s ==================

$ grep -rn --include='*.py' -E 'b64decode|"=" \* \(-len' tldw_Server_API/app/core/MCP_unified \
    | grep -v "/tests/" | grep urlsafe
modules/implementations/prompts_catalog.py:91:    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
modules/implementations/prompts_catalog.py:116:        raw = base64.urlsafe_b64decode((raw_cursor + padding).encode("ascii"))
modules/implementations/filesystem_receipts.py:83:        return base64.urlsafe_b64encode(_canonical_json(envelope)).decode("ascii")
modules/implementations/filesystem_receipts.py:91:            envelope_bytes = base64.urlsafe_b64decode(receipt.encode("ascii"))
```

## Findings

```
FINDING mcp-unified-4
  axis:        correctness
  class:       n/a
  severity:    High
  sites:       pyproject.toml:638 (testpaths includes the in-app tree);
               .github/workflows/ci.yml:1758-1760 (and the identical shard at :3260, :4630,
               :5923, :7222) — the `platform-mcp-core` shard lists only
               tldw_Server_API/tests/MCP and tldw_Server_API/tests/MCP_unified;
               .github/workflows/backend-required.yml:193-195 — required gate runs
               tldw_Server_API/tests/unit only;
               .github/workflows/coverage-required.yml:154-157 — required gate runs
               tldw_Server_API/tests/unit + sanity_tests only;
               .github/workflows/mcp-unified-rc.yml:15-26 — the only workflow naming any
               in-app test file, and it names 11 of 147 as *path triggers*, additionally
               admission-gated on vars.LICENSE_FIRST_CI_ENABLED == 'true' (:43-45);
               the orphaned tree: tldw_Server_API/app/core/MCP_unified/tests/ — 147 files,
               90,042 LOC, 63.6% of the module's total line count.
  canonical:   NONE
  destination: n/a
  knowledge:   "which tests gate a change to MCP_unified". `pyproject.toml` says one thing
               (both trees), the CI shard map says another (Tree B only), and nothing
               reconciles them.
  scenario:    Confirmed by a live red test. A developer running bare `pytest` picks up the
               in-app tree via testpaths:638 and sees
               `test_filesystem_module.py::test_filesystem_glob_marks_file_size_unavailable`
               fail. CI never runs that file — `grep -c` over ci.yml returns 0 — so the
               failure has survived on HEAD since the test was added in 5009fc8b95 on
               2026-06-03, roughly three and a half months. Generalising: any regression in
               the 105 in-app files that test `core.MCP_unified` (including
               test_filesystem_module.py at 3,677 LOC, test_git_module.py at 1,928,
               test_skills_module.py at 1,566, test_kanban_module.py at 1,266,
               test_prompts_catalog.py at 1,164) merges green. The 35 gateway files are
               partially covered by mcp-unified-rc.yml, but only when one of 11 specific
               paths changes *and* the LICENSE_FIRST_CI_ENABLED repo variable is set — so a
               change to `core/MCP_unified/modules/` triggers neither workflow's MCP tests.
  impact:      High: 63.6% of the module's lines are a test suite that no contractual gate
               (Docs/Development/CI_REQUIRED_GATES.md names backend-required,
               security-required, coverage-required, frontend-required, e2e-required,
               container-build-check) executes, and the proof that this matters is already
               on the branch in the form of a months-old red test. This is also the direct
               answer to the module's boundary question: the in-app tests/ location is not a
               deliberate boundary, it is a place where tests fell out of CI's path map.
  cost-driver: n/a
  tests:       n/a — this finding is about the tests.
  effort:      cheap for the highest-value slice: add
               `tldw_Server_API/app/core/MCP_unified/tests` to the `platform-mcp-core` shard
               `paths` in ci.yml. Moderate overall, because doing so will surface the red
               test (fix it first, per finding mcp-unified-5) and will add the 35
               distribution-building gateway files to a shard that does not install build
               tooling — those should be split out or excluded by marker in the same change.
  owner-only:  no (.github/ is not on CONTRIBUTING.md's owner-only list)
  confidence:  confirmed (grep -c returns 0; both required gates' test selections read
               directly; the red test reproduced)
```

```
FINDING mcp-unified-12
  axis:        encapsulation
  class:       n/a
  severity:    Medium
  sites:       apps/mcp-unified/ — 125 files, 46,521 LOC, its own pyproject.toml, LICENSE,
               README.md, USER_GUIDE.md and pytest-artifact-gate.ini, and **no tests/ directory**;
               its test suite lives at tldw_Server_API/app/core/MCP_unified/tests/ — 35 files,
               39,533 LOC, identified by `^(import|from) mcp_unified`;
               the coupling is hardcoded by path arithmetic:
               tests/test_runtime_package_boundary.py:35-44 (`REPO_ROOT = parents[5]`,
               `STANDALONE_PROJECT_ROOT = REPO_ROOT / "apps" / "mcp-unified"`) and
               tests/test_extraction_contracts.py:16-17 (same shape);
               tests/test_gateway_fastapi_package.py:25-27 likewise;
               stale contradiction: tldw_Server_API/app/core/MCP_unified/README.md:3 —
               "the standalone package/gateway is planned but not shipped in this tree yet".
  canonical:   NONE
  destination: apps/mcp-unified/tests/ — the package's own suite, alongside its own
               pyproject.toml, with the server-side extraction ratchets
               (test_extraction_contracts.py, test_runtime_package_boundary.py) staying in
               the server tree because they legitimately span both.
  knowledge:   "what ships with the mcp-unified distribution and how a consumer verifies it".
               ADR-033's Consequences require the release to "prove typed IDs, arbitrary JSON
               roots, all new limits, moving cursors, error metadata, direct dependency
               packaging, and both native/fallback binary stdio paths" — the tests that prove
               all of that are not in the distributable.
  scenario:    n/a (encapsulation axis). Concretely: `pip install mcp-unified` yields a
               package whose 46,521 LOC has zero accompanying tests; a downstream consumer
               (ADR-033 names Chatbook) cannot run the conformance suite without cloning
               tldw_server. Conversely, every one of those 35 test files reaches five
               directories up (`parents[5]`) into the server repo, so the package's suite
               cannot move without rewriting its path arithmetic — the coupling is
               structural, not incidental.
  impact:      Medium: it does not break anything today, and the extraction ratchets are
               genuinely good work (they are the two highest-churn files in the module,
               which is the signature of a boundary being actively maintained rather than
               neglected). It is filed because the placement is what causes finding 4 —
               39,533 LOC of tests for a package that CI's server shards have no reason to
               run sit in a directory CI's server shards do not run — and because
               README.md:3 tells the next reader the package does not exist.
  cost-driver: n/a
  tests:       the subject. The ratchets themselves
               (test_extraction_contracts.py 4,429 LOC / 43 commits,
               test_runtime_package_boundary.py 1,669 LOC / 50 commits) are in scope and
               **downgrade the risk on the extraction's correctness**; they do not address
               placement. Secondary observation: test_runtime_package_boundary.py:29 sets
               `pytestmark = pytest.mark.unit` while the module builds sdists and wheels and
               shells out — so `pytest -m unit` picks up a distribution build. That marker is
               wrong and would bite immediately if finding 4's fix routed the tree into the
               `unit`-marked required gate.
  effort:      moderate — mechanical file moves plus rewriting `parents[5]` path arithmetic
               in 35 files, and it changes which workflow owns them. Needs a short design
               record (`Docs/Design/2026-MM-DD-mcp-unified-test-ownership-design.md`) naming
               which suite each file belongs to and how mcp-unified-rc.yml's trigger list
               changes; no ADR, since ADR-033 already settles the package boundary itself and
               this only relocates its verification.
  owner-only:  no (apps/mcp-unified/** is not on CONTRIBUTING.md's owner-only list — only
               apps/tldw-frontend, apps/extension and apps/packages/ui are)
  confidence:  confirmed (no tests/ dir; the 35-file/39,533-LOC count; the parents[5] path
               arithmetic; the stale README line)
```

```
FINDING mcp-unified-13
  axis:        duplication
  class:       true-duplication
  severity:    Low
  sites:       modules/implementations/prompts_catalog.py:encode_prompt_cursor (66-91) —
               `base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")` at :91 over
               `json.dumps(payload, sort_keys=True, separators=(",", ":"))` at :89;
               modules/implementations/prompts_catalog.py:decode_prompt_cursor (94-160) —
               `padding = "=" * (-len(raw_cursor) % 4)` at :115, `urlsafe_b64decode` at :116,
               `_CURSOR_VERSION` check at :120-123 (`_CURSOR_VERSION = 1` at :41).
               This is this module's only participant in repo-wide cluster C1 (22 sites).
  canonical:   NONE repo-wide.
  destination: a new `core/Utils/pagination_cursor.py` owning exactly one responsibility:
               encode/decode an **opaque, unauthenticated** pagination cursor (versioned
               JSON, canonical separators, max-bytes guard, strippped padding). It must NOT
               absorb the signed/crypto sites in C1 (core/AuthNZ/api_key_crypto.py,
               api/v1/endpoints/notes.py signature segments) — flattening an opaque cursor
               and a signed token into one helper is the failure mode to avoid, and this
               module has one of each: prompts_catalog is opaque, while
               modules/implementations/filesystem_receipts.py:83,91 is an integrity-bearing
               receipt envelope and must stay separate.
               Explicitly NOT core/Utils/Utils.py and NOT core/http_client.py.
  knowledge:   the opaque-cursor wire format: canonical JSON, urlsafe base64, stripped
               padding, a version field, and the decode-side validation that a cursor from a
               different version or a moved catalog is rejected rather than misread.
  scenario:    n/a (duplication axis). Change amplification for this module specifically is
               low — one site, self-contained, well tested — which is why the severity is
               Low. It is filed so that the repo-wide C1 consolidation has this module's site
               enumerated and correctly classified rather than swept in with the signed
               tokens. Note the adjacent site in the same package, filesystem_receipts.py:83,
               does **not** strip padding and therefore needs no repad idiom; that asymmetry
               inside one package is the small, local instance of the wider inconsistency.
  impact:      Low: a single site, no divergence inside this module, strong local tests.
  cost-driver: n/a
  tests:       app/core/MCP_unified/tests/test_prompts_catalog.py (1,164 LOC) — covers the
               cursor round trip, the version check, the identifier-pair invariant and the
               config_index invariant; app/core/MCP_unified/tests/test_mcp_prompts_http.py
               and tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py cover the HTTP
               surface. Well covered, which makes any future extraction cheap.
               (Import-grep reachability, not measured coverage.)
  effort:      cheap for this site once a destination exists; the destination decision itself
               is a repo-wide call that belongs to the C1 owner, not to this module.
  owner-only:  no
  confidence:  confirmed (the site and its idiom); assumption (that the repo-wide C1
               consolidation will happen at all — this module can equally be left alone)
```

## Suggested Refactor/Actions

1. **Add `tldw_Server_API/app/core/MCP_unified/tests` to the `platform-mcp-core` shard in
   `.github/workflows/ci.yml:1758-1760`** (and the four identical shard blocks at `:3260`,
   `:4630`, `:5923`, `:7222`). Sequence it *after* the `_glob_paths` fix from stage 2 so the
   shard does not land red. Exclude or marker-gate the 35 distribution-building gateway
   files in the same change — they need build tooling the shard does not install, and
   `mcp-unified-rc.yml` already owns them.
   Do this **first**; it is the cheapest change in this whole review and it is what makes
   every other finding's fix verifiable.
2. **Fix the `pytest.mark.unit` marker on `test_runtime_package_boundary.py:29`** before (1)
   lands anywhere near a `-m unit` selection. A test that builds sdists and wheels is not a
   unit test, and `backend-required.yml:194` selects on exactly that marker.
3. **Move the 35 `mcp_unified`-importing files into `apps/mcp-unified/tests/`** and let the
   package own its own suite, keeping `test_extraction_contracts.py` and
   `test_runtime_package_boundary.py` on the server side because they legitimately span both
   trees. Needs the short design record named in finding 12; no ADR.
4. **Correct `tldw_Server_API/app/core/MCP_unified/README.md:3`.** One line, and it currently
   tells every new reader that 46,521 LOC of shipped package does not exist.
5. **Do not create Backlog tasks from this ledger** — the briefing makes Backlog the ledger of
   record, managed via its MCP/CLI, and this review is read-only. Propose four tasks:
   (a) CI shard gap + marker fix [findings 4, 12-marker], (b) sanitizer consolidation
   [findings 1, 2], (c) `BaseModule` accessors + admin-claims alignment [findings 3, 6, 8],
   (d) small independents [findings 5, 7, 9, 10, 11]. Each should link the stage file that
   produced it.
6. **Base branch assumption:** `CONTRIBUTING.md:86,121` says PRs target `dev` while
   `origin/HEAD` resolves to `main`. Nothing in this review depends on the choice; the CI
   path additions in (1) apply to both since `mcp-unified-rc.yml:4-7` already lists both
   branches.
