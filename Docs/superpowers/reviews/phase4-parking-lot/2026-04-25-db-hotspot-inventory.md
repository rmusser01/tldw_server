# Phase 4.3 DB Hotspot Inventory

**Date:** 2026-04-25

**Status:** Inventory complete; per-file implementation plans pending.

## Purpose

Rank the remaining large DB modules before any decomposition work. This is a planning artifact only. It records static size, test signals, first safe boundaries, and files to avoid while Phase 2/3 closeout is still unstable.

## Method

Static line-count snapshot from:

```bash
wc -l tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/Kanban_DB.py tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/app/core/DB_Management/Guardian_DB.py tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/app/core/DB_Management/Evaluations_DB.py tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/Workflows_DB.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
```

This inventory did not build a call graph or run tests.

## Hotspot Ranking

| File | Lines | Test signal | First safe boundary | Recommendation |
| --- | ---: | --- | --- | --- |
| `ChaChaNotes_DB.py` | 32696 | Many direct ChaCha and DB management tests | None without a dedicated plan | Defer. Too broad for the first Phase 4.3 extraction. |
| `Kanban_DB.py` | 8626 | Kanban tests plus RAG kanban retriever coverage | Read-only row mapping or query helpers | Later candidate after transaction inventory. |
| `PromptStudioDatabase.py` | 7181 | Prompt Studio API and DB tests | Job/test result row mapping | Later candidate; jobs coupling raises risk. |
| `Collections_DB.py` | 5377 | Collections test suite exists | None in this workspace | Avoid in this branch because active dirty work already exists here. |
| `Guardian_DB.py` | 3297 | Guardian tests exist | Validation and row mapping helpers | Defer until security-sensitive behavior has explicit focused tests. |
| `Watchlists_DB.py` | 3178 | Watchlists tests exist | Scheduler-independent query helpers | Defer until scheduler/jobs coupling is mapped. |
| `Evaluations_DB.py` | 2943 | Evaluations and unified CRUD tests exist | Eval result row mapping or batch helper extraction | Good candidate if the goal is eval schema cleanup. |
| `Prompts_DB.py` | 2881 | Prompt Management legacy and new tests exist | Prompt row mapping, import/export helper boundaries | Best first candidate for a low-risk extraction plan. |
| `Workflows_DB.py` | 2871 | Workflows and Postgres returning tests exist | Read-only query helpers only | Defer because scheduler/workflow execution coupling is high. |
| `ManuscriptDB.py` | 2471 | Writing/manuscript tests exist | Manuscript row conversion helpers | Possible later candidate after UI/workflow callers are checked. |

## Test Inventory Signals

Direct or nearby coverage exists for each major hotspot family:

- ChaCha: `tldw_Server_API/tests/ChaChaNotesDB/` and `tldw_Server_API/tests/DB_Management/test_chacha_*`
- Kanban: `tldw_Server_API/tests/kanban/` and `tldw_Server_API/tests/RAG_NEW/unit/test_kanban_retriever.py`
- Prompt Studio: `tldw_Server_API/tests/prompt_studio/`
- Collections: `tldw_Server_API/tests/Collections/`
- Guardian: `tldw_Server_API/tests/Guardian/`
- Watchlists: `tldw_Server_API/tests/Watchlists/`
- Evaluations: `tldw_Server_API/tests/Evaluations/` and `tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py`
- Prompts: `tldw_Server_API/tests/Prompt_Management/` and `tldw_Server_API/tests/Prompt_Management_NEW/`
- Workflows: `tldw_Server_API/tests/Workflows/` and `tldw_Server_API/tests/DB_Management/test_postgres_returning_and_workflows.py`
- Manuscripts: `tldw_Server_API/tests/Writing/test_manuscript_*`

## Recommended First Target

Start with `Prompts_DB.py` after Phase 2/3 closeout is stable.

Rationale:

- It is large enough to benefit from decomposition but not one of the highest-risk files.
- Prompt Management has legacy and newer test coverage.
- The likely first boundary is mechanical row mapping or import/export helpers rather than transaction behavior.
- It avoids `Collections_DB.py`, which is already dirty in this workspace.

Alternate first target:

- `Evaluations_DB.py`, if maintainers want Phase 4.3 to support eval work first.

Draft per-file plan:

- `Docs/superpowers/plans/2026-04-25-phase4-3-prompts-db-decomposition-plan.md`

## Required Per-File Plan Before Code Movement

For the chosen file, create a dedicated plan with:

- public method inventory
- internal helper inventory
- transaction boundary inventory
- schema and migration dependency inventory
- current test list and focused test command
- rollback plan
- Bandit touched-scope command

## Do Not Do Yet

- Do not start with `ChaChaNotes_DB.py`.
- Do not touch `Collections_DB.py` in this workspace unless the existing dirty work is intentionally adopted.
- Do not move transaction boundaries and row mapping in the same first slice.
- Do not change SQL semantics without focused regression tests for sqlite and Postgres paths where applicable.

## Handoff Checklist

- [ ] Maintainers accept `Prompts_DB.py` or choose the alternate first file.
- [x] A per-file plan is created before runtime edits.
- [x] Focused tests are listed before code movement.
- [x] Transaction boundaries are documented before helper extraction.
- [x] Touched-scope Bandit command is identified before PR handoff.
