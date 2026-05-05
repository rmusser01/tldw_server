---
id: TASK-60
title: Implement native CodeGraph C# extractor slice
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 03:25'
updated_date: '2026-05-05 03:35'
labels:
  - codegraph
  - mcp
  - csharp
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1277'
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next Stage 5 native CodeGraph language slice after the merged Java/Kotlin work: conservative C# extractor support with namespace, using directive, type, constructor, method, property, and same-file simple call capture where practical. Keep the slice narrow: no C/C++, no Roslyn semantic model, no project/solution analysis, no full overload/type resolution, and no Jobs/file-watcher changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 C# files are promoted from planned metadata to foundation metadata with dependency_missing and symbol_extraction reflecting optional parser availability.
- [x] #2 C# extraction indexes namespace, using/import, class/interface/struct/enum/record, constructor, method, and property symbols with deterministic IDs and conservative unresolved references for external imports and receiver calls.
- [x] #3 The indexer safely skips C# files when the optional C# parser is unavailable without blocking other extractable languages or foreground limits.
- [x] #4 MCP search can find indexed C# type and member symbols after codegraph.index in a fixture workspace.
- [x] #5 Focused loader, registry, extractor, indexer, and MCP tests cover C# behavior plus dependency-missing paths, with Ruff, Bandit on touched production scope, and git diff --check passing before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented focused C# CodeGraph slice. Verified tree-sitter-c-sharp 0.23.5 exposes tree_sitter_c_sharp.language and installed tree-sitter-c-sharp>=0.23,<0.24 into the shared venv for local tests.

Added optional dependency probing, parser loader mapping, csharp foundation metadata, CodeGraphIndexer registration, CSharpTreeSitterExtractor, extractor/indexer/registry/loader/MCP search tests, and implementation plan Docs/superpowers/plans/2026-05-05-native-codegraph-csharp-extractor-implementation-plan.md.

Verification: focused RED tests failed for unsupported csharp or missing extractor before implementation; focused C# regression set passed with 9 passed; full CodeGraph plus MCP focused suite passed with 117 passed and 5 warnings; Ruff passed on touched CodeGraph/MCP/test scope; Bandit JSON at /tmp/bandit_codegraph_csharp.json reported results 0 and errors 0; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the native CodeGraph C# extractor slice. C# is now a dependency-aware foundation language backed by optional tree-sitter-c-sharp, with conservative extraction for using directives, namespaces, classes, interfaces, structs, records, enums, constructors, methods, properties, same-file simple calls, and unresolved external/receiver references. Known limits remain intentional: no Roslyn project model, project/solution analysis, partial type merging, overload/type inference, inheritance, extension-method, or cross-file semantic resolution.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
