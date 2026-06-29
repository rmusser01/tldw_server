# Pre-main UAT Evidence

- Run id: `pre-main-uat-20260629054510`
- Task id: `TASK-12064`
- Status: Not Started
- Evidence root: `Docs/Product/WebUI/evidence/pre_main_uat/pre-main-uat-20260629054510`
- Raw root: `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510`

## Source Control State

Captured after creating the UAT execution Backlog task and before adding the evidence Markdown files.

```text
## codex/pr1982-ci-fanout-fixes
?? "backlog/tasks/task-12064 - Execute-pre-main-UAT-matrix-for-PR-1982.md"
?? tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md
?? tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md
```

The untracked watchlist templates above were present before this Task 1 evidence shell and are intentionally excluded from this run commit:

- `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`
- `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`

## Commit State

```text
a290e5e4283af9420175640b75534f68acd83fd0
```

## PR State

```json
{"baseRefName":"main","headRefName":"dev","headRefOid":"70aa7b3e92edcf3b6310a77a0f9bf48f18c55b02","mergeStateStatus":"UNSTABLE","number":1982,"url":"https://github.com/rmusser01/tldw_server/pull/1982"}
```

## Notes

- Disposable raw fixtures and runtime profiles are under `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510` and are not committed.
- `uat.env` under the raw root exports the run variables for later UAT tasks.
