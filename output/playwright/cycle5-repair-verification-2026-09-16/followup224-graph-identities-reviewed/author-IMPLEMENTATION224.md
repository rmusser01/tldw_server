# UAT224 / TASK13260.162 — normalize Graph note identities

The native administrator response returns a raw note UUID and its real tag edge. The workspace and Canvas focus use `note:<id>`; the selected node is therefore absent, and Relationships falsely says no relationships are visible. Existing backend GraphService constructs note nodes with raw IDs and emits/prunes edges against the nodes in each page. Tag/source nodes already have their own identities. This is independent of UAT210 mapping and UAT221 permission/cache handling.

The bounded design keeps the public backend API unchanged and normalizes at the existing frontend `normalizeGraph` boundary after strict validation. A map contains only known, unprefixed note nodes. Their node IDs and matching source/target endpoints receive `note:`. Already-prefixed note IDs, edge IDs, tag/source IDs, metadata, cursors and request note IDs remain unchanged. Unknown endpoints are not guessed. No workspace, hook, cache, permission or suggestion lifecycle edits belong to this unit.

Only the two files in owned-manifest.json are owned. Snapshot copies and an isolated owned.patch are retained. The backend service implementation and existing three consumers were inspected before the change. The five new tests exercise the actual service (transport doubled) and real relationship grouping. They catch missing selection/relationships, both endpoint directions, mixed identities, input mutation, idempotence, cursor-page handling and unrelated/missing endpoints.

- Causal RED before production edit: 3 failed / 2 passing controls / zero skips.
- Final scoped GREEN: 75 passed / 4 suites / zero skips, including existing suggestion service, graph hook and workspace view suites.
- Scoped ESLint and whitespace checks passed. Full compiler and Bandit attempt receipts are retained separately; Bandit cannot analyze TypeScript and must not be represented as meaningful TS coverage.
- Original native evidence: `.tmp/uat198-181-native-20260917/admin221-graph-events.txt`, `admin224-relationships-open.txt`, `admin221-graph.png`.
- Native acceptance and independent review remain pending. No data mutation, permissions change or full matrix run is claimed.

Reproduce from apps/tldw-frontend: `node_modules/.bin/vitest run ../packages/ui/src/services/tldw/__tests__/note-graph-identities.test.ts ../packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.view-mode.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx`.
