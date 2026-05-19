# Native CodeGraph Java/Kotlin Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Stage 5 Java/Kotlin native CodeGraph extractor slice without broadening into C#, C/C++, Jobs mode, or full type/classpath resolution.

**Architecture:** Keep language parsing in focused Tree-sitter extractors under `tldw_Server_API/app/core/CodeGraph/extractors/`, with shared JVM-family helpers only where they remove real duplication. Keep `CodeGraphIndexer` as the dependency-aware wiring point and `CodeGraphModule` unchanged except for search visibility through existing repository APIs.

**Tech Stack:** Python 3.11, pytest, optional `tree-sitter`, `tree-sitter-java`, `tree-sitter-kotlin`, SQLite CodeGraph repository, existing Unified MCP CodeGraph module.

---

## Scope

Included:

- Java package/import/type/method/constructor extraction.
- Kotlin package/import/class/object/interface/function extraction.
- Conservative same-file calls resolved by simple name, with other calls/imports stored as unresolved refs.
- Dependency-aware registry/indexer wiring and MCP search coverage through existing tools.
- Focused verification and TASK-49 closeout.

Excluded:

- C#, C, and C++ extractors.
- Full Java classpath, Gradle/Maven, Kotlin compiler, type inference, inheritance resolution, overloaded method resolution, or cross-file JVM symbol resolution.
- New MCP tools, Jobs-backed indexing, file watching, or source-context behavior changes.

## File Map

- Modify: `pyproject.toml`
  - Add optional `.[codegraph]` parser dependencies for Java/Kotlin after a local parser smoke check.
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py`
  - Add parser module mappings for `java` and `kotlin`.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/jvm_common.py`
  - Shared node text, descendant traversal, qualified-name, call-site, and node/edge construction helpers for Java/Kotlin.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/java_extractor.py`
  - Java Tree-sitter extraction.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/kotlin_extractor.py`
  - Kotlin Tree-sitter extraction.
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/__init__.py`
  - Export Java/Kotlin extractor classes if needed by local import patterns.
- Modify: `tldw_Server_API/app/core/CodeGraph/language_registry.py`
  - Promote Java/Kotlin from planned metadata to foundation metadata with dependency-aware `symbol_extraction`.
- Modify: `tldw_Server_API/app/core/CodeGraph/indexer.py`
  - Register Java/Kotlin extractors only when their parsers are available.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py`
  - Cover parser loading and missing dependency behavior.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py`
  - Cover Java/Kotlin dependency-aware metadata.
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_java_extractor.py`
  - Java extractor behavior.
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_kotlin_extractor.py`
  - Kotlin extractor behavior.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`
  - Indexer wiring/search coverage for Java/Kotlin.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
  - Existing MCP search coverage for indexed Java/Kotlin symbols.
- Modify: `backlog/tasks/task-49 - Implement-native-CodeGraph-Java-Kotlin-extractor-slice.md`
  - Keep task state, verification, and final summary current.

## Task 1: Dependency Gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py`

- [x] **Step 1: Install/verify candidate parser dependencies**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pip install "tree-sitter-java>=0.23,<0.24" "tree-sitter-kotlin>=1.1,<1.2"
```

Expected: packages install into the shared repo venv. If installation fails due network or wheel availability, stop and report the blocker rather than replacing this slice with regex parsing.

- [x] **Step 2: Write RED loader tests**

Add tests that call `load_parser("java")` and `load_parser("kotlin")`, parse compact Java/Kotlin snippets, and assert missing-package behavior can report `tree_sitter_java` / `tree_sitter_kotlin` without raising.

- [x] **Step 3: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py -q
```

Expected before implementation: fail with unsupported Tree-sitter language for Java/Kotlin.

- [x] **Step 4: Implement parser mappings**

Add:

```python
"java": ("tree_sitter_java", "language"),
"kotlin": ("tree_sitter_kotlin", "language"),
```

to `_LANGUAGE_MODULES`.

Add dependency bounds to `pyproject.toml` only after the smoke test confirms the import/function names.

- [x] **Step 5: Run GREEN loader tests**

Run the same pytest command. Expected: all loader tests pass.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml \
  tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py
git commit -m "feat: add codegraph jvm parser loading"
```

## Task 2: Shared JVM Extraction Helpers

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/jvm_common.py`
- Test indirectly through Java/Kotlin extractor tests in Tasks 3 and 4.

- [x] **Step 1: Add a small helper module**

Include only helpers shared by both languages:

- byte-backed `node_text(source, node) -> str`
- `named_descendants_of_type(node, *types) -> iterator`
- line/column conversion from Tree-sitter points to 1-based line and column
- `JvmCallSite` dataclass
- helper for creating `CodeGraphNode` with `make_node_id()`
- helper for resolving call sites by simple local callable name

- [x] **Step 2: Keep helpers language-neutral**

Do not encode Java/Kotlin grammar node-type decisions in this file unless both languages use the same Tree-sitter node type and semantics.

- [x] **Step 3: Defer commit**

Commit helpers with the first extractor that uses them so the branch stays buildable.

## Task 3: Java Extractor

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/java_extractor.py`
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_java_extractor.py`
- Create/modify: `tldw_Server_API/app/core/CodeGraph/extractors/jvm_common.py`

- [x] **Step 1: Write RED Java extractor test**

Test source:

```java
package com.example.app;

import java.util.List;
import com.example.tools.Helper;

public class Greeter {
    public Greeter() {
        setup();
    }

    public String greet(String name) {
        return helper(name);
    }

    private String helper(String value) {
        return value.toUpperCase();
    }
}
```

Expected nodes:

- module node for `src/main/java/com/example/app/Greeter.java`
- package node `com.example.app`
- import nodes for `java.util.List` and `com.example.tools.Helper`
- class node `com.example.app.Greeter`
- constructor node `com.example.app.Greeter.Greeter`
- method nodes `com.example.app.Greeter.greet` and `com.example.app.Greeter.helper`

Expected relationships:

- constructor has a resolved same-file call edge to `setup` only if `setup` exists; in this fixture it should be unresolved.
- `greet` has a resolved same-file call edge to `helper`.
- imports are unresolved refs unless the target file is resolved in a later slice.

- [x] **Step 2: Run RED Java test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_java_extractor.py -q
```

Expected: fail because `JavaTreeSitterExtractor` does not exist.

- [x] **Step 3: Implement minimal Java extractor**

Implement:

- dependency errors through `load_parser("java")`
- UTF-8 decode errors as `ExtractionResult(errors=(...))`
- parse errors as `ExtractionResult(errors=("Java parse error",))`
- package declarations from `package_declaration`
- imports from `import_declaration`
- classes/interfaces/enums/records from type declarations where straightforward
- constructors and methods under classes
- same-file simple call resolution from `method_invocation` names inside method/constructor bodies

Do not attempt overload or receiver type resolution. Receiver calls such as `client.fetch()` should become unresolved refs.

- [x] **Step 4: Run GREEN Java test**

Run the same pytest command. Expected: pass.

- [x] **Step 5: Add deterministic ID test**

Add one test extracting the same Java source twice and assert node IDs are identical.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/CodeGraph/extractors/jvm_common.py \
  tldw_Server_API/app/core/CodeGraph/extractors/java_extractor.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_java_extractor.py
git commit -m "feat: add codegraph java extractor"
```

## Task 4: Kotlin Extractor

**Files:**
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/kotlin_extractor.py`
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_kotlin_extractor.py`
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/jvm_common.py` only for genuinely shared needs.

- [x] **Step 1: Write RED Kotlin extractor test**

Test source:

```kotlin
package com.example.app

import com.example.tools.Helper
import kotlin.collections.List

class Greeter {
    fun greet(name: String): String {
        return helper(name)
    }

    private fun helper(value: String): String {
        return value.uppercase()
    }
}

object Registry {
    fun create(): Greeter {
        return Greeter()
    }
}
```

Expected nodes:

- module node for `src/main/kotlin/com/example/app/Greeter.kt`
- package node `com.example.app`
- import nodes for both imports
- class node `com.example.app.Greeter`
- function/method nodes `com.example.app.Greeter.greet`, `com.example.app.Greeter.helper`
- object node `com.example.app.Registry`
- function node `com.example.app.Registry.create`

Expected relationships:

- `greet` has a resolved same-file call edge to `helper`.
- `create` may leave `Greeter` constructor call unresolved unless constructor nodes are explicitly modeled for Kotlin in this slice.

- [x] **Step 2: Run RED Kotlin test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_kotlin_extractor.py -q
```

Expected: fail because `KotlinTreeSitterExtractor` does not exist.

- [x] **Step 3: Implement minimal Kotlin extractor**

Implement:

- dependency errors through `load_parser("kotlin")`
- UTF-8 decode errors and parse errors
- package/import declarations
- class/object/interface declarations
- named functions, including functions inside class/object bodies
- same-file simple function call resolution

Do not model Kotlin overloads, extensions, generics, delegated properties, or compiler semantics.

- [x] **Step 4: Run GREEN Kotlin test**

Run the same pytest command. Expected: pass.

- [x] **Step 5: Add deterministic ID test**

Add one test extracting the same Kotlin source twice and assert node IDs are identical.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/CodeGraph/extractors/kotlin_extractor.py \
  tldw_Server_API/app/core/CodeGraph/extractors/jvm_common.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_kotlin_extractor.py
git commit -m "feat: add codegraph kotlin extractor"
```

## Task 5: Registry And Indexer Wiring

**Files:**
- Modify: `tldw_Server_API/app/core/CodeGraph/language_registry.py`
- Modify: `tldw_Server_API/app/core/CodeGraph/indexer.py`
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/__init__.py`
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py`
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`

- [x] **Step 1: Write RED registry test**

Update registry tests so Java/Kotlin are foundation-stage language entries with `symbol_extraction=True` only when their parser packages are present.

Expected before implementation: Java/Kotlin still report `stage == "planned"`.

- [x] **Step 2: Write RED indexer test**

Add a test with `Service.java` and `Greeter.kt`, then assert:

- index status is `complete`
- both files have `node_count > 0`
- `repo.search_nodes("Greeter", limit=10)` returns Java/Kotlin symbols as appropriate

- [x] **Step 3: Implement registry and indexer wiring**

Move Java/Kotlin out of `_PLANNED_LANGUAGES` into `_foundation_languages()` with dependency-missing checks:

```python
java_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_java"))
kotlin_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_kotlin"))
```

In `CodeGraphIndexer.__init__`, register extractors only when `load_parser("java").available` / `load_parser("kotlin").available`.

In `_extract()`, pass `workspace_root` to Java/Kotlin only if extractors need it. Do not use build-system globals.

- [x] **Step 4: Run focused registry/indexer tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py \
  -q
```

Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/CodeGraph/language_registry.py \
  tldw_Server_API/app/core/CodeGraph/indexer.py \
  tldw_Server_API/app/core/CodeGraph/extractors/__init__.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py
git commit -m "feat: wire codegraph java kotlin extractors"
```

## Task 6: MCP Search Regression

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

- [x] **Step 1: Write RED MCP search test**

Add a test that creates Java/Kotlin files in a temp workspace, runs `codegraph.index`, then calls existing `codegraph.search` for a Java/Kotlin symbol.

Expected before wiring: search returns no Java/Kotlin symbol because files were planned-language skipped or no extractor was registered.

- [x] **Step 2: Run RED MCP test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py::test_codegraph_search_finds_java_kotlin_symbols_after_index -q
```

Expected: fail before implementation.

Observed after Task 5 wiring: passed without MCP implementation changes because the existing module delegates to the newly wired registry/indexer/repository path.

- [x] **Step 3: Ensure existing MCP code needs no new tool changes**

The MCP module should already surface Java/Kotlin through `status`, `index`, `files`, `search`, `node`, and context/impact tools. Only change MCP implementation if validation blocks known language IDs after registry changes.

- [x] **Step 4: Run GREEN MCP test**

Run the same pytest command. Expected: pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "test: cover mcp java kotlin codegraph search"
```

## Task 7: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-49 - Implement-native-CodeGraph-Java-Kotlin-extractor-slice.md`

- [x] **Step 1: Run focused CodeGraph/MCP tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  -q
```

Expected: all pass.

- [x] **Step 2: Run Ruff**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
```

Expected: `All checks passed!`

- [x] **Step 3: Run Bandit**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_java_kotlin.json
```

Expected: JSON has `errors: []` and `results: []`.

- [x] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [x] **Step 5: Update TASK-49**

Record:

- dependency versions installed/verified
- RED/GREEN evidence
- verification command results
- known limits: no classpath/type/build-system resolution

- [x] **Step 6: Commit task finalization**

```bash
git add "backlog/tasks/task-49 - Implement-native-CodeGraph-Java-Kotlin-extractor-slice.md"
git commit -m "docs: finalize codegraph java kotlin task"
```

## Review Risks

- Tree-sitter JVM package APIs may not match the JS-family `language()` shape. Verify before pinning.
- Kotlin grammar node names may differ from expected names; inspect parsed trees in a scratch REPL if RED tests fail for grammar-shape reasons.
- Java/Kotlin call resolution must stay conservative. Do not link receiver/member calls across objects unless the target is unambiguously same-file and simple-name local.
- Keep Java/Kotlin dependency absence non-fatal: the module must remain importable and status should explain missing parser packages.
