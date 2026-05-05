# Native CodeGraph C# Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Stage 5 C# native CodeGraph extractor slice without broadening into C/C++, Roslyn project analysis, Jobs mode, or full type resolution.

**Architecture:** Follow the Java/Kotlin slice shape: keep C# parsing in a focused Tree-sitter extractor under `tldw_Server_API/app/core/CodeGraph/extractors/`, wire optional dependency availability through the centralized parser loader and language registry, and rely on existing repository/MCP search paths. Keep the extractor truthful and conservative: symbols and same-file simple calls are indexed, while external imports and receiver calls remain unresolved references.

**Tech Stack:** Python 3.11, pytest, optional `tree-sitter`, optional `tree-sitter-c-sharp>=0.23,<0.24`, SQLite CodeGraph repository, existing Unified MCP CodeGraph module.

---

## Scope

Included:

- C# `using` directive, namespace, type, constructor, method, and property extraction.
- Types: class, interface, struct, enum, and record declarations.
- Conservative same-file simple invocation resolution by method/constructor name.
- Receiver/member-access calls and external imports stored as unresolved refs.
- Dependency-aware registry/indexer wiring and MCP search coverage through existing tools.
- Focused verification and TASK-60 closeout.

Excluded:

- C and C++ extractors.
- Roslyn, `.csproj`/`.sln`, NuGet, source generators, partial type merging, overload resolution, inheritance/interface implementation resolution, cross-file semantic type resolution, or extension-method resolution.
- New MCP tools, Jobs-backed indexing, file watching, or context/impact behavior changes.

## File Map

- Modify: `pyproject.toml`
  - Add optional `.[codegraph]` parser dependency for C# after local parser smoke verification.
- Modify: `tldw_Server_API/app/core/CodeGraph/dependencies.py`
  - Add `tree_sitter_c_sharp` to optional language dependency probing.
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/tree_sitter_loader.py`
  - Add parser module mapping for `csharp`.
- Create: `tldw_Server_API/app/core/CodeGraph/extractors/csharp_extractor.py`
  - C# Tree-sitter extraction.
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/__init__.py`
  - Export `CSharpTreeSitterExtractor` if local import patterns require it.
- Modify: `tldw_Server_API/app/core/CodeGraph/language_registry.py`
  - Promote C# from planned metadata to foundation metadata with dependency-aware `symbol_extraction`.
- Modify: `tldw_Server_API/app/core/CodeGraph/indexer.py`
  - Register the C# extractor only when its parser is available.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py`
  - Cover C# parser loading and missing dependency behavior.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py`
  - Cover C# dependency-aware metadata.
- Create: `tldw_Server_API/tests/CodeGraph/test_codegraph_csharp_extractor.py`
  - C# extractor behavior.
- Modify: `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`
  - Indexer wiring and dependency-missing coverage for C#.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
  - Existing MCP search coverage for indexed C# symbols.
- Modify: `backlog/tasks/task-60 - Implement-native-CodeGraph-C-extractor-slice.md`
  - Keep task state, verification, and final summary current.

## Stage 1: Dependency Gate

**Goal:** Verify and wire the optional C# parser package.
**Success Criteria:** Loader can parse a compact C# fixture, missing-package behavior reports `tree_sitter_c_sharp`, and dependency bounds are recorded.
**Tests:** `tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py`
**Status:** Complete

- [x] **Step 1: Verify candidate parser package**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pip index versions tree-sitter-c-sharp
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pip install "tree-sitter-c-sharp>=0.23,<0.24"
```

Observed: PyPI exposes `tree-sitter-c-sharp 0.23.5`; local package exposes `tree_sitter_c_sharp.language`.

- [x] **Step 2: Write RED loader tests**

Add tests that call `load_parser("csharp")`, parse a compact C# class fixture, and assert missing-package behavior can report `tree_sitter_c_sharp` without raising.

- [x] **Step 3: Run RED loader tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_tree_sitter_loader.py -q
```

Expected before implementation: fail with unsupported Tree-sitter language for C#.

- [x] **Step 4: Implement parser mapping and dependency probe**

Add:

```python
"csharp": ("tree_sitter_c_sharp", "language"),
```

to `_LANGUAGE_MODULES`, add `tree_sitter_c_sharp` to optional dependency probing, and add `tree-sitter-c-sharp>=0.23,<0.24` to `.[codegraph]`.

- [x] **Step 5: Run GREEN loader tests**

Run the same pytest command. Expected: all loader tests pass.

## Stage 2: C# Extractor

**Goal:** Extract conservative C# symbols and same-file calls.
**Success Criteria:** Fixture extraction captures namespace, using directives, types, constructors, methods, properties, same-file call edges, unresolved receiver calls, parse errors, and deterministic node IDs.
**Tests:** `tldw_Server_API/tests/CodeGraph/test_codegraph_csharp_extractor.py`
**Status:** Complete

- [x] **Step 1: Write RED C# extractor tests**

Test source should include:

```csharp
using System;
using Collections = System.Collections.Generic;

namespace Example.App;

public class Greeter {
    public Greeter() { Setup(); }
    public string Name { get; set; }
    public string Greet(string name) { return Helper(name); }
    private string Helper(string value) { return value.ToUpperInvariant(); }
}

internal interface IMarker { void Mark(); }
public record Person(string Name);
public struct Point { public int X { get; set; } }
public enum Mode { Basic, Advanced }
```

Expected nodes:

- module node for `src/Example/App/Greeter.cs`
- namespace node `Example.App`
- import nodes for `System` and `Collections = System.Collections.Generic`
- class/interface/record/struct/enum nodes with visibility
- constructor, method, and property nodes under their type scopes

Expected relationships:

- `Greet` has a resolved same-file call edge to `Helper`
- constructor call to missing `Setup` is unresolved
- receiver call `value.ToUpperInvariant()` is unresolved as `value.ToUpperInvariant`
- using directives are unresolved refs

- [x] **Step 2: Run RED extractor tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_csharp_extractor.py -q
```

Expected: fail because `CSharpTreeSitterExtractor` does not exist.

- [x] **Step 3: Implement minimal C# extractor**

Implement:

- dependency errors through `load_parser("csharp")`
- UTF-8 decode errors as `ExtractionResult(errors=(...))`
- parse errors as `ExtractionResult(errors=("C# parse error",))`
- `using_directive` as import nodes preserving alias syntax
- `namespace_declaration` and `file_scoped_namespace_declaration`
- `class_declaration`, `interface_declaration`, `struct_declaration`, `enum_declaration`, and `record_declaration`
- `constructor_declaration`, `method_declaration`, and `property_declaration` inside type declaration bodies
- same-file simple invocation resolution from `invocation_expression` where the function is an `identifier`
- unresolved call refs where the function is a `member_access_expression`

Do not attempt overload, receiver type, or cross-file resolution.

- [x] **Step 4: Run GREEN extractor tests**

Run the same pytest command. Expected: pass.

## Stage 3: Registry And Indexer Wiring

**Goal:** Make C# a dependency-aware foundation language and index C# graph rows only when extractable.
**Success Criteria:** C# appears as foundation metadata, missing parser metadata is visible, indexer persists graph rows when available, and safely skips C# when unavailable.
**Tests:** `tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py`, selected `test_codegraph_indexer.py`
**Status:** Complete

- [x] **Step 1: Write RED registry/indexer tests**

Add assertions that:

- `CodeGraphLanguageRegistry(...).language_for_path("Program.cs")` returns `csharp`
- C# stage is `foundation` when promoted
- `dependency_missing == ("tree_sitter_c_sharp",)` and `symbol_extraction is False` when the package is missing
- indexer skips non-extractable C# without tripping foreground limits
- indexer persists C# graph rows when the parser is available

- [x] **Step 2: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py::test_indexer_extracts_csharp_graph_rows_during_index \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py::test_indexer_does_not_count_non_extractable_csharp_files_against_foreground_limits -q
```

Expected: fail until registry/indexer wiring exists.

- [x] **Step 3: Implement registry/indexer wiring**

Move C# out of `_PLANNED_LANGUAGES`, add C# foundation metadata, and register `CSharpTreeSitterExtractor` in `CodeGraphIndexer` when `load_parser("csharp").available` is true.

- [x] **Step 4: Run GREEN registry/indexer tests**

Run the same pytest command. Expected: pass.

## Stage 4: MCP Search Coverage

**Goal:** Prove existing MCP CodeGraph tools can search indexed C# symbols.
**Success Criteria:** `codegraph.index` followed by `codegraph.search` finds C# type/member symbols in a fixture workspace.
**Tests:** `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
**Status:** Complete

- [x] **Step 1: Write RED MCP search test**

Add or extend a CodeGraph MCP test fixture with `Greeter.cs`, then assert search finds `Greeter`, `Greet`, or `Name` after indexing.

- [x] **Step 2: Run RED MCP test**

Run the focused test. Expected: fail until indexer wiring and extractor are complete.

- [x] **Step 3: Implement any missing MCP exposure**

Prefer no MCP production changes if the existing module delegates correctly through registry/indexer/repository.

- [x] **Step 4: Run GREEN MCP test**

Run the focused test. Expected: pass.

## Stage 5: Verification And Closeout

**Goal:** Verify the full focused surface and record task completion.
**Success Criteria:** Focused tests, Ruff, Bandit, and diff check pass; TASK-60 records verification and known limits.
**Tests:** Full focused CodeGraph/MCP command below.
**Status:** Complete

- [x] **Step 1: Run focused suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q
```

- [x] **Step 2: Run Ruff**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
```

- [x] **Step 3: Run Bandit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_csharp.json
```

Inspect the JSON and record `errors` / `results`.

- [x] **Step 4: Run diff check**

```bash
git diff --check
```

- [x] **Step 5: Update TASK-60**

Record verification, known limits, and final summary. Check acceptance criteria and DoD only after verification passes.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml Docs/superpowers/plans/2026-05-05-native-codegraph-csharp-extractor-implementation-plan.md \
  backlog/tasks/task-60\ -\ Implement-native-CodeGraph-C-extractor-slice.md \
  tldw_Server_API/app/core/CodeGraph tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: add codegraph csharp extractor"
```
