# Repeatable Comprehensive Repository Audit Process

This process reruns the comprehensive audit style used for the 2026-06-27 repository audit. It produces review artifacts only; production and runtime source files stay unchanged unless the user explicitly redirects the work from audit to remediation.

## Prerequisites And Safety Gates

- Start from an approved audit design and implementation plan, or get explicit user approval for the audit scope before creating artifacts.
- Work in one clean audit worktree created from latest `origin/dev`. Do not audit a dirty checkout as the primary target.
- Create or reuse one Backlog.md execution task before repository file edits begin.
- Keep all audit outputs under `Docs/superpowers/reviews/<date>-repo-audit/` and the matching Backlog task file.
- Agents may inspect source and tests, but they do not edit production code during the audit.
- Dependency installs, networked package audits, Docker pulls or builds, service startup, and environment-changing setup require coordinator approval and must be recorded in evidence.
- Redact secrets, tokens, sensitive environment values, and sensitive local data from evidence logs.

## Worktree, Branch, And Backlog Setup

1. Fetch `origin/dev`.
2. Record `git rev-parse origin/dev` as the audit baseline.
3. Create a single worktree and branch, using names that include the audit date.
4. Verify the worktree is clean before scaffold edits.
5. Search for an existing active Backlog execution task. Reuse it only when it clearly covers this exact audit execution.
6. If no matching execution task exists, create one with acceptance criteria for baseline recording, domain completion, finding normalization, high and critical validation, and final report delivery.
7. Record the task ID, baseline SHA, network-refresh status, worktree path, and branch name in `inventory.md` and `evidence/command-log.md`.

## Baseline Freshness Gate

The coordinator owns baseline freshness. Before dispatching agents or publishing final findings:

- Run `git fetch origin dev` when network permission is available.
- Compare the recorded baseline with `git rev-parse origin/dev`.
- If `origin/dev` has not advanced, keep the recorded baseline.
- If `origin/dev` has advanced and the audit branch has only audit-artifact edits, rebase the audit branch onto `origin/dev`, record the previous baseline as superseded, update all active baseline references to the refreshed SHA, and keep the branch HEAD distinct from the baseline SHA.
- If rebase conflicts occur, stop for coordinator handling. Do not discard audit scaffold edits.
- If fetch is unavailable, ask the user whether to proceed with the local `origin/dev` baseline. When approved, mark the baseline `not network-refreshed` in `inventory.md` and `final-report.md`.

After a refresh, record these facts in `evidence/command-log.md`: fetch command, clean status before refresh edits, rebase command, superseded baseline label, refreshed baseline SHA, and current audit branch HEAD.

## Artifact Scaffold Requirements

Create this artifact tree:

- `inventory.md`
- `findings-index.json`
- `final-report.md`
- `remediation-backlog-draft.md`
- `evidence/command-log.md`
- `domains/authnz-admin.md`
- `domains/media-ingestion-storage.md`
- `domains/chat-rag-llm.md`
- `domains/jobs-scheduler-workflows.md`
- `domains/mcp-sandbox-agent-protocol.md`
- `domains/db-migrations-data-durability.md`
- `domains/webui-extension-api-contracts.md`
- `domains/integrations-providers.md`
- `domains/ci-deployment-operations-release.md`
- `specialists/security-boundaries.md`
- `specialists/reliability-lifecycle.md`
- `specialists/api-webui-contracts.md`
- `specialists/test-coverage-verification.md`
- `specialists/dependency-static-analysis.md`

Each domain and specialist report starts with scope, baseline, report owner, findings table, index mapping guidance, confirmed issues, likely risks, improvement opportunities, files inspected, tests or scans run, blocked or unverified areas, and evidence notes.

## Inventory Collection

Build shared evidence before domain dispatch:

- Endpoint decorators under `tldw_Server_API/app/api/v1/endpoints`.
- Backend test file inventory under `tldw_Server_API/tests`.
- Frontend API-client usage across existing app and extension paths.
- Dependency manifests and lockfiles.
- DB migration and SQL surfaces.
- CI, deployment, Docker, operations, and sample configuration files.
- Production-scope Bandit summary when available.
- Prior audit and remediation references relevant to the current scope.

Write inventory outputs under `evidence/`, then update `inventory.md` with evidence file paths, line counts, skipped paths, and known limitations.

## Domain Agent Dispatch

Use no more than four domain agents at a time. Recommended batches:

- Batch 1: AuthNZ and Admin; DB, Migrations, and Data Durability; WebUI, Extension, and API Contracts; CI, Deployment, Operations, and Release Surfaces.
- Batch 2: Media, Ingestion, and Storage; Chat, RAG, and LLM; Jobs, Scheduler, and Workflows; Integrations and Providers.
- Batch 3: MCP, Sandbox, and Agent Protocol.

Each domain agent receives:

- Audit worktree path.
- Domain scope and primary source paths.
- Allowed write path limited to its report and scoped evidence files.
- Rule to avoid production-code edits.
- Rule to use existing local tooling only unless coordinator approval is granted.
- Candidate ID prefix such as `CANDIDATE-authnz-admin-001`.
- Requirement to record inspected files, commands run, blockers, residual risk, and evidence.

Run a merge checkpoint after each batch. A domain report is incomplete if it lacks a scope statement, inspected files, tests or scans run, or explicit blocked/unverified entries.

## Finding Normalization

The coordinator normalizes domain candidates into `findings-index.json` after domain review:

- Merge duplicates under one stable ID.
- Use IDs in the form `AUDIT-YYYY-MM-DD-<DOMAIN>-NNN`.
- Keep refuted candidates in source reports with disposition notes, outside the accepted findings index.
- Require each accepted finding to include `id`, `title`, `severity`, `confidence`, `category`, `evidence_tier`, `evidence_strength`, `status`, `owner_domain`, `source_report`, `evidence`, `affected_paths`, `recommendation`, and `validation_status`.
- Validate JSON after every index edit with `jq empty` or `python -m json.tool`.

Static-analysis output is evidence only. It becomes an accepted finding only after source review confirms relevance.

## Specialist Review Passes

Use no more than three specialist agents at a time. Recommended batches:

- Batch 1: Security boundaries; Reliability and async lifecycle; API and WebUI contract drift.
- Batch 2: Test coverage and verification gaps; Dependency and static-analysis risk.

Specialists read `findings-index.json`, `inventory.md`, all completed domain reports, and targeted code paths. They confirm, refute, escalate, cross-link, or add missing cross-domain evidence. They create a new candidate only when the issue is absent from domain outputs.

## Coordinator Validation Rules

Before final publication:

- Fully re-read every high and critical finding, its source report, affected paths, code path, impact statement, evidence strength, and remediation recommendation.
- Mark high and critical findings as validated only after this re-read.
- Demote, refute, or move unsupported high and critical claims to improvement opportunities.
- Sample medium and low findings when confidence is not high or evidence conflicts.
- Reject scan-only claims as confirmed issues.
- Record validation decisions in `final-report.md`, `findings-index.json`, and the relevant source report when disposition changes.

## Final Report And Remediation Backlog

The final report includes:

- Executive summary.
- Baseline SHA and network-refresh status.
- Severity-ranked accepted findings.
- High and critical coordinator validation table.
- Confirmed issues, likely risks, and improvement opportunities.
- Coverage gaps and explicit unverified scope.
- Verification notes and evidence references.

The remediation backlog draft groups follow-up slices by priority, finding IDs, owner domain, reviewable slice, dependencies, suggested verification, and notes. It remains a draft until the user approves actual Backlog task creation.

## Verification And Static Analysis

Final audit verification includes:

- Filler-token scan over the audit artifact directory.
- Required domain and specialist report counts.
- JSON validation for `findings-index.json`.
- `git diff --check`.
- `git status --short`.

For audit-only documentation changes, record that Bandit is not applicable because no production code changed. When the audit or a later remediation touches code, run Bandit over the touched production scope from the project virtual environment and fix new findings in changed code before completion.

## Commit And Staging Notes

- Commit review artifacts separately from design and planning artifacts when possible.
- Stage only audit artifact files and the matching Backlog task file for audit-documentation commits.
- The repository ignores `*.json`, so force-add `Docs/superpowers/reviews/<date>-repo-audit/findings-index.json` when staging with Git.
- Run whitespace and JSON checks before committing.
- Do not use bypass flags for hooks.
- If the branch was rebased for a baseline refresh, record both the refreshed baseline SHA and the current audit branch HEAD.

## Backlog Marker Handling

Backlog MCP and CLI may fail to find a task in a worktree or may rewrite task sections incorrectly. If the tooling cannot preserve task sections, edit the task file directly only with user approval. After direct edits, verify the task file contains exactly one `SECTION:FINAL_SUMMARY:BEGIN` marker and exactly one `SECTION:FINAL_SUMMARY:END` marker.
