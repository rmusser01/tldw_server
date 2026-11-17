# Core-owned chat wire contracts — TASK-13643

Pure history and native fork/retention model definitions now live under core/Chat. Shared frozen Resume contracts and the character completion result live under core/Character_Chat. API schema modules directly re-export the existing names; eight core callers import their models from core. The architecture ratchet removes only two resolved legacy baseline entries and adds no exemption.

Definition bodies were preserved during relocation. New-file optional annotations and one forward return annotation use equivalent future-annotation style; validators, defaults and strict/frozen behavior remain. Standard import sorting preserves all non-import AST. Both independent bounded source/API reviews found no actionable issue, lost export or cycle.

The canonical full exporter succeeds before and after import sorting: openapi.json, committed fingerprint and path union are byte-identical (2129paths,3301schemas). Connected history, projection, fork transaction/commit, asset preparation and Roleplay contract/completion suites pass661tests with no skips,1896runtimewarnings,639.68s and process exit0, including actual PostgreSQL Roleplay qualification. Architecture controls pass3/3 after causal1failure/2passes. Scoped production Bandit is0findings/scanerrors and matches the parent; scoped Ruff is clean after7import-sort fixes.

Normal client compilers and exact-file hooks are recorded in the task. This change enforces the existing dependency direction in [Architecture](../Architecture.md); ADR required:no, no new durable rule, API shape or persistence policy. Whole-PR findings, final independent review, final remote-dev rebase and merge remain open. Installed-extension/physical AT acceptance is not claimed.

PR: https://github.com/rmusser01/tldw_server/pull/3229
