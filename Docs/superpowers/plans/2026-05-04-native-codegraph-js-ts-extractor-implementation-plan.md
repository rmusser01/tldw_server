# Native CodeGraph JS/TS Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Stage 3 native CodeGraph JavaScript/TypeScript extractor slice with conservative symbols, imports/exports, same-file calls, and trusted-workspace path-alias resolution.

**Architecture:** Keep extraction in focused core modules under `tldw_Server_API/app/core/CodeGraph/extractors/`, keeping the MCP module thin. JS/TS parsing should use the optional `.[codegraph]` Tree-sitter dependency set when available, while the module remains importable when those dependencies are missing. Import resolution should be workspace-root bounded and conservative: resolve relative and configured path-alias targets under the trusted workspace, record external/unresolved imports, and avoid TypeScript compiler/type-check requirements.

**Tech Stack:** Python 3.11, pytest, SQLite repository helpers, optional `tree-sitter`, `tree-sitter-javascript`, `tree-sitter-typescript`, Unified MCP CodeGraph module.

---

## Scope

Implement this slice only:

- JavaScript/TypeScript/TSX/JSX extractor modules.
- A small Tree-sitter loader/helper scoped to CodeGraph.
- Relative import and `tsconfig.json` / `jsconfig.json` path-alias resolution.
- Indexer wiring and focused MCP/search regression coverage.

Do not implement:

- `codegraph.context` or `codegraph.impact`.
- Jobs/background indexing.
- C, C++, C#, Java, or Kotlin extractors.
- Full TypeScript compiler integration or project type checking.
- `node_modules` indexing.

## File Structure

- Create: `tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py`
  - Optional dependency import boundary and parser construction helpers.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/js_ts_imports.py`
  - Workspace-bounded relative import and path-alias resolution helpers.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/javascript_extractor.py`
  - JavaScript/JSX extraction plus shared JS-family graph builder if practical.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/typescript_extractor.py`
  - TypeScript/TSX parser selection and TS-specific declarations.
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_js_ts_imports.py`
  - Relative and alias resolver tests.
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_javascript_extractor.py`
  - JS/JSX extractor tests.
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_typescript_extractor.py`
  - TS/TSX extractor tests.
- Modify: `tldw_Server_API/app/core/CodeGraph/indexer.py`
  - Register JS/TS extractors when optional dependencies are available and pass workspace context needed for import resolution.
- Modify: `tldw_Server_API/app/core/CodeGraph/language_registry.py`
  - Mark JS/TS symbol extraction truthfully when dependencies are available.
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/__init__.py`
  - Export extractor classes if local pattern requires it.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`
  - Replace the inventory-only JS/TS expectation with graph extraction behavior under injected extractors/dependencies.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
  - Add one MCP-level smoke proving JS/TS symbols are searchable after indexing.
- Modify: `backlog/tasks/task-38 - Implement-native-CodeGraph-JS-TS-extractor-slice.md`
  - Keep notes, acceptance criteria, verification, and final summary current.

## Dependency Gate

The current shared venv does not have `tree_sitter`, `tree_sitter_javascript`, or `tree_sitter_typescript` installed. Before implementing parser-dependent production code, run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python - <<'PY'
import importlib.util
for name in ["tree_sitter", "tree_sitter_javascript", "tree_sitter_typescript"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

If missing, install or verify the optional extra:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pip install -e ".[codegraph]"
```

Expected parser matrix after install:

- `tree-sitter>=0.25,<0.26`
- `tree-sitter-javascript>=0.25,<0.26`
- `tree-sitter-typescript>=0.23,<0.24`

If installation is unavailable in the environment, stop and report the blocker rather than replacing this slice with regex parsing.

## Task 1: Parser Loader And Smoke Tests

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py`

- [x] **Step 1: Write RED tests for dependency-aware parser loading**

Test cases:

- Missing modules return an unavailable result without raising at module import time.
- JavaScript parser can parse `export function helper() { return 1; }`.
- TypeScript parser can parse `interface User { id: string }`.
- TSX parser can parse `export function Card() { return <div />; }`.

- [x] **Step 2: Run tests and verify RED**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py -q
```

Expected: fails because `tree_sitter_loader.py` does not exist.

- [x] **Step 3: Implement loader**

Create small value objects such as:

```python
@dataclass(frozen=True)
class ParserLoadResult:
    parser: Any | None = None
    missing: tuple[str, ...] = ()
    error: str | None = None
```

Use dynamic imports inside functions so normal CodeGraph imports work without `.[codegraph]`.

- [x] **Step 4: Run loader tests and commit**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py -q
git add tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py
git commit -m "test: add codegraph tree-sitter loader"
```

## Task 2: Import Resolver

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/js_ts_imports.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_js_ts_imports.py`

- [x] **Step 1: Write RED resolver tests**

Cover:

- `./utils` resolves to `src/utils.ts` when imported from `src/app.ts`.
- `../shared/button` resolves with `.tsx` extension.
- `@/components/Button` resolves through `tsconfig.json` paths.
- `~/*`, `@web/*`, and `@tldw/ui/*` style aliases resolve when targets stay under the workspace root.
- Alias targets that escape the workspace root are ignored and reported as unresolved.
- External packages like `react` are classified as external/unresolved, not resolved into `node_modules`.

- [x] **Step 2: Run tests and verify RED**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_js_ts_imports.py -q
```

- [x] **Step 3: Implement resolver**

Implement focused helpers:

- `load_js_ts_project_config(workspace_root: Path, source_path: str)`.
- `resolve_js_ts_import(workspace_root: Path, source_path: str, specifier: str)`.
- `resolve_relative_import(...)`.
- `resolve_path_alias_import(...)`.

Use JSON parsing only. Strip JSONC comments only if tests require real repo config parsing; otherwise start with valid JSON fixtures and document JSONC as follow-up if needed.

- [x] **Step 4: Run resolver tests and commit**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_js_ts_imports.py -q
git add tldw_Server_API/app/core/CodeGraph/extractors/js_ts_imports.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_js_ts_imports.py
git commit -m "feat: add codegraph js ts import resolver"
```

## Task 3: JavaScript Extractor

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/javascript_extractor.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_javascript_extractor.py`

- [x] **Step 1: Write RED extractor tests**

Cover:

- Module node for `src/app.js`.
- Function declaration node.
- Arrow-function variable node.
- Class and method nodes.
- React-like component node for PascalCase function returning JSX.
- Import and re-export nodes.
- Same-file call edge for direct identifier calls.
- Member expression call unresolved ref, not false-linked to a same-file symbol.
- Parse errors return `ExtractionResult(errors=...)`.

- [x] **Step 2: Run tests and verify RED**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_javascript_extractor.py -q
```

- [x] **Step 3: Implement conservative graph builder**

Use Tree-sitter named nodes and a small traversal wrapper. Prefer node type checks over broad text regex. Use stable IDs with `make_node_id()` and `make_edge_id()`, matching Python extractor conventions.

Minimum node kinds:

- `module`
- `function`
- `component`
- `class`
- `method`
- `import`

Minimum edge/ref kinds:

- `calls`
- unresolved `call`
- unresolved `import`

- [x] **Step 4: Run tests and commit**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_javascript_extractor.py -q
git add tldw_Server_API/app/core/CodeGraph/extractors/javascript_extractor.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_javascript_extractor.py
git commit -m "feat: add codegraph javascript extractor"
```

## Task 4: TypeScript And TSX Extractor

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/typescript_extractor.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_typescript_extractor.py`

- [x] **Step 1: Write RED TypeScript tests**

Cover:

- Function, class, method.
- Interface, type alias, enum.
- TSX component function.
- Import/export capture.
- Deterministic IDs.

- [x] **Step 2: Run tests and verify RED**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_typescript_extractor.py -q
```

- [x] **Step 3: Implement TypeScript extractor**

Reuse JavaScript traversal where possible. Keep TS-specific declarations small and explicit. Do not attempt type resolution.

- [x] **Step 4: Run tests and commit**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_typescript_extractor.py -q
git add tldw_Server_API/app/core/CodeGraph/extractors/typescript_extractor.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_typescript_extractor.py
git commit -m "feat: add codegraph typescript extractor"
```

## Task 5: Indexer Wiring And Search Regression

**Files:**
- Modify: `tldw_Server_API/app/core/CodeGraph/indexer.py`
- Modify: `tldw_Server_API/app/core/CodeGraph/language_registry.py`
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

- [x] **Step 1: Write RED indexer tests**

Cover:

- `.ts` files now produce graph nodes when dependencies are available.
- `.tsx` component is searchable through repository search.
- JS/TS extraction failure becomes `extraction_failed` without aborting indexing.
- Missing parser dependencies keep module importable and status truthful.

- [x] **Step 2: Run tests and verify RED**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

- [x] **Step 3: Wire extractors**

Instantiate JS/TS extractors in `CodeGraphIndexer.__init__` only when loader health says the parser is available. Pass workspace root to the extractor or resolver through an explicit extraction context rather than global state.

- [x] **Step 4: Run focused tests and commit**

```bash
python -m pytest tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q
git add tldw_Server_API/app/core/CodeGraph/indexer.py \
  tldw_Server_API/app/core/CodeGraph/language_registry.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: wire codegraph js ts extractors"
```

## Task 6: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-38 - Implement-native-CodeGraph-JS-TS-extractor-slice.md`

- [x] **Step 1: Run focused regression tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q
```

- [x] **Step 2: Run Ruff on touched files**

```bash
python -m ruff check tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
```

- [x] **Step 3: Run Bandit on touched production scope**

```bash
python -m bandit -r tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_js_ts_extractor.json
```

Expected: `results 0`, `errors 0`, or documented pre-existing/non-actionable findings outside touched code.

- [x] **Step 4: Run whitespace check**

```bash
git diff --check
```

- [x] **Step 5: Update Backlog task**

Mark acceptance criteria and Definition of Done items complete, record verification output, and add a final summary.

- [x] **Step 6: Commit closeout**

```bash
git add backlog/tasks/task-38\ -\ Implement-native-CodeGraph-JS-TS-extractor-slice.md
git commit -m "docs: finalize codegraph js ts extractor task"
```

## Known Risks

- Tree-sitter package APIs can differ across minor versions. Keep the loader small and verify against the pinned optional-extra matrix before production wiring.
- `tsconfig.json` commonly contains JSONC comments. If real repo configs require JSONC parsing, add a tiny comment stripper with tests or defer with a documented unresolved reason; do not add a large config parser dependency in this slice.
- Cross-file import resolution should not imply symbol-level target accuracy yet. Store import nodes and unresolved refs with candidate file paths rather than claiming fully resolved imported symbols.
- React component detection should be conservative. PascalCase plus JSX return is enough; avoid overclassifying ordinary PascalCase functions without JSX evidence.
