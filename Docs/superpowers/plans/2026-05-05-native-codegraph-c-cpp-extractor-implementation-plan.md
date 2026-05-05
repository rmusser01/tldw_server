# Native CodeGraph C/C++ Extractor Implementation Plan

## Stage 1: Parser and Metadata Wiring
**Goal**: Add dependency-aware C and C++ parser support to the CodeGraph foundation language set.
**Success Criteria**: `tree_sitter_c` and `tree_sitter_cpp` are optional dependencies, the loader supports `c` and `cpp`, and registry metadata reports C/C++ as foundation languages with `dependency_missing` when parser packages are absent.
**Tests**: Loader parser tests, missing dependency tests, registry metadata tests.
**Status**: Complete

## Stage 2: Conservative Extractors
**Goal**: Add Tree-sitter extraction for common C and C++ symbols without compiler-semantic claims.
**Success Criteria**: C extraction captures includes, structs, enums, functions, and same-file simple calls. C++ extraction captures includes, namespaces, classes, structs, enums, methods, and same-file simple calls. Parse errors and deterministic IDs are covered.
**Tests**: `test_codegraph_c_extractor.py`, `test_codegraph_cpp_extractor.py`.
**Status**: Complete

## Stage 3: Indexer Integration
**Goal**: Register C and C++ extractors only when optional parser packages are available.
**Success Criteria**: Indexing persists graph rows for C/C++ when dependencies exist and skips C/C++ files without consuming foreground file limits when dependencies are missing.
**Tests**: C/C++ graph-row indexer test and dependency-missing foreground-limit test.
**Status**: Complete

## Stage 4: MCP Visibility
**Goal**: Expose indexed C/C++ symbols through existing CodeGraph MCP search.
**Success Criteria**: `codegraph.index` indexes C/C++ fixture files and `codegraph.search` finds C functions and C++ methods by language/kind filters.
**Tests**: C/C++ MCP search roundtrip test.
**Status**: Complete

## Stage 5: Verification and PR
**Goal**: Prove the slice is clean enough for review and open a PR against `dev`.
**Success Criteria**: Focused CodeGraph/MCP suite, Ruff, Bandit on touched production scope, and `git diff --check` pass. Backlog TASK-63 records final verification and PR link.
**Tests**: Full focused CodeGraph/MCP suite plus static and security checks.
**Status**: Complete
