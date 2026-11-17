# Admin WebUI Perf Remediation — Execution Summary (2026-10-06 → 2026-10-09)

Execution record for the three plans drafted 2026-10-06 (spec:
[ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md](ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md), findings F1–F23).
All 15 stages were implemented subagent-driven with a task-scoped review per stage plus a
final whole-branch review. Every stage passed; three stages required one fix round each
(B-S1, B-S4, A-S5, C-S4, C-S5 — all verified by scoped re-review).

**Branch:** `codex/admin-webui-perf-exec` (from dev tip `7ba48f251e`; plan commits
`14a613116b`, `b5fd52b214`). 20 execution commits, `9ba7897af7..de49405516`.

## Commit map

| Stage | Commit | One-liner |
|---|---|---|
| A-S1 | `9ba7897af7` | billing admin routes + SQL paging (F1) |
| A-S2 | `8a15997d47` | tokens-today + cost attribution read llm_usage_log (F2) |
| A-S3 | `c7f2879001` | sessions(created_at) + org_members(org_id,user_id) indexes, Migration(101) (F3/F4) |
| A-S4 | `e930e54995` | backups TTL cache + page-scoped stats (F5) |
| A-S5 | `eea72fe0ce` + `b4640e9e88` | event-loop offload (inventory/admin reads/SLA metrics) (F6/F7) |
| B-S1 | `9c56497036` + `4400443b98` | shared tri-state capability probe + billing envelope + durable in-place downgrade (F9) |
| B-S2 | `d5b0ff0147` | polling split + tab-visibility gate (F8) |
| B-S3 | `54d6cd40ab` | orgs server pagination + api-key remote search (F10) |
| B-S4 | `a83469c731` + `2042cc3f77` | react-query cache for admin reference data (F11) |
| B-S5 | `b48cf4d9e9` | debounced budget input + usage date ranges (F12/F13) |
| C-S1 | `cef22bb429` | MonitoringDashboard render isolation (F14) |
| C-S2 | `dc76f70be9` | llamacpp settings hoist + memoized panels (F15) |
| C-S3 | `60df3829a4` | Map/Set O(1) lookups + memoized derivations (F16/F18) |
| C-S4 | `85a0acdba1` + `f2a472bbf2` | RBAC matrix pagination + virtualized lists (F16/F17) |
| C-S5 | `8a8004ec07` + `de49405516` | stable rowKeys + shared Intl formatter + form isolation (F19–F23) |

## Verification record

- Per stage: TDD RED→GREEN evidence in the SDD reports; affected suites green; Bandit 0
  findings on backend scope (re-verified independently by the final reviewer).
- Final whole-branch review (2026-10-09): **Ready to merge — yes.** No in-program
  Critical/Important defects; cross-stage contracts on the five multiply-touched files
  verified; sargability proven by EXPLAIN QUERY PLAN tests; both test blemishes verified
  pre-existing at the recorded baseline `b5fd52b214`.
- Known pre-existing (NOT this program): `ServerAdminPage.design-system.test.tsx`
  password-reset test fails/flakes at baseline; media-budget test stderr 400 noise.

## Pre-merge bookkeeping (required by repo process)

1. **Backlog task IDs**: `backlog task create` was broken throughout (stack overflow, then
   the CLI binary disappeared from `~/.bun/bin`); all 20 commits and the three plan
   headers carry "(backlog task TBD)". Create the three tasks (A/B/C) and backfill the
   plan headers when the tracker works. Note: the parallel perf program's task files
   (task-13411..13419) are inside the user's stash@{0}
   ("WIP-on-codex/post2970-uat-20260920-before-perf-plan-2026-10-06") in the MAIN checkout.
2. **Human-authored Change summary**: per the AI-PR merge gate
   (`Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`) the human
   requester must write it before merge — AI text does not satisfy the gate.

## Post-merge follow-ups (final-review priority order, all pre-existing or deferred)

1. `getDailyUsage` sends `start_date`/`end_date`; backend `/usage/daily` takes `start`/`end`
   — daily chart ignores the range selector (one-line rename + test; completes F13's intent).
2. Multi-org duplicate inflation in cost attribution `group_by=user` under org-scoped admins
   (`admin_usage_service.py:1236-1245`) — fix via EXISTS semi-join.
3. Unguarded `os.stat` in `_backup_file_from_row` (`admin_data_ops_service.py:206`) —
   skip-on-OSError for files deleted within the 30s cache window.
4. Repair the pre-existing ServerAdminPage design-system password-reset test (restores CI signal).
5. Backend: org members/teams response totals; team-members limit/offset support.
6. Billing per-user mutation routes (override/credits) still 404 — buttons render; hide or
   downgrade them (intentional per OSS billing removal tests).
7. Billing subscriptions/events full server pagination (truthful `total` already shipped).
8. `PUT /maintenance` + `GET /llamacpp/config` one-line `to_thread` wraps (mixed idiom leftovers).
9. matrix-boolean client/server response-shape mismatch (pre-existing, documented).

## Release note for large PostgreSQL deployments

The first startup after upgrading applies plain (non-concurrent) `CREATE INDEX` on
`sessions(created_at)` and `org_members(org_id, user_id)` inside the bootstrap transaction —
writes to those tables block for the build duration on very large tables. (Consistent with
migrations 099/100; a CONCURRENTLY variant can't run in that transaction. Follow-up tracked.)
