# Backlog.md Task Tracking Design

Date: 2026-05-03
Owner: Codex collaboration session
Status: Approved after user review, pending implementation planning

## Summary

Adopt Backlog.md as the required project task-tracking layer for development work in this repository.

Every future effort that changes repository files should be tied to a Backlog.md task. The task should preserve the operational history that git commits do not capture by themselves: why the work exists, what plan was followed, what files were touched, what verification ran, what was skipped, and how the work ended.

This does not replace the existing superpowers workflow. Backlog.md becomes the kanban and historical work-item system. Superpowers specs, implementation plans, review loops, test-driven development, and verification gates continue to apply where the repo process already requires them.

## References

- Backlog.md repository: `https://github.com/MrLesk/Backlog.md`
- Official Backlog.md agent guidance: `https://github.com/MrLesk/Backlog.md/blob/main/src/guidelines/mcp/agent-nudge.md`
- Backlog.md CLI reference: `https://github.com/MrLesk/Backlog.md/blob/main/CLI-INSTRUCTIONS.md`

## Goals

1. Initialize Backlog.md for `/Users/macbook-dev/Documents/GitHub/tldw_server2` if it is not already initialized.
2. Follow Backlog.md's official MCP-first guidance for AI-agent usage.
3. Update root `AGENTS.md` with a dedicated Backlog.md task-tracking policy.
4. Require a Backlog.md task before edits for any work that changes repository files.
5. Preserve existing superpowers-driven design, planning, implementation, review, testing, security, and commit practices.
6. Keep Backlog.md task history useful by recording plans, notes, acceptance criteria, verification, blockers, PR links, and final summaries.
7. Avoid duplicate task records by requiring search-first behavior before creating new tasks.
8. Keep git history under normal repo control by leaving Backlog.md auto-commit disabled.

## Non-Goals

1. Replacing git commits, PRs, GitHub issues, or superpowers specs/plans.
2. Rewriting every historical plan or issue into Backlog.md during initial adoption.
3. Forcing read-only investigation to create a task when no repository files are changed.
4. Manually editing Backlog.md task markdown files as the normal workflow.
5. Enabling Backlog.md hook bypass or any workflow that uses `--no-verify`.
6. Creating board export snapshots as a routine generated artifact.
7. Updating every agent instruction file unless the user explicitly approves broader instruction synchronization.

## Required Policy

Root `AGENTS.md` should state that Backlog.md is mandatory for repo-changing development work.

The rule should be:

- Any work that changes repository files must have an associated Backlog.md task before file edits begin.
- This includes code, tests, docs, config, scripts, tracked generated artifacts, cleanup edits, and agent-instruction changes.
- Read-only investigation can proceed without a task.
- If investigation turns into edits, the agent must stop, find or create a Backlog.md task, and then continue.
- Agents should work one Backlog.md task at a time where practical.

Backlog.md is the status and history layer. It should link to richer artifacts instead of duplicating them wholesale. For example, a task can link to a GitHub issue, PR, `Docs/superpowers/specs/...`, `Docs/superpowers/plans/...`, review document, or relevant commit.

## Official Workflow Alignment

The repo instructions should follow Backlog.md's official MCP-first advice:

1. If the Backlog.md MCP server is available, read the workflow overview before creating, executing, or finalizing tasks.
2. Prefer the official MCP resources/tools for task creation, task execution, and task finalization.
3. If MCP resources are unavailable but Backlog.md tools are available, use the official instruction fallback.
4. If MCP is not configured but the CLI is installed, use CLI commands as the fallback.
5. If neither MCP nor CLI is available, pause before repo file edits unless the user explicitly approves a temporary exception.

The initial `AGENTS.md` wording should not attempt to reproduce the entire official Backlog.md workflow. It should direct agents to the official workflow resource first, then document this repo's additional requirements. Current Backlog.md docs reference `backlog://workflow/overview` in the agent guidance and `backlog://docs/task-workflow` in the README; the implementation should follow whichever resource the installed Backlog.md MCP server exposes, then fall back to `backlog.get_backlog_instructions()` if needed.

## CLI Fallback

The CLI fallback should use Backlog.md commands rather than manual task-file edits.

Useful commands to mention in `AGENTS.md`:

```bash
backlog search "query" --plain
backlog task list --plain
backlog task <id> --plain
backlog task create "Title" -d "Description" --ac "Acceptance criterion"
backlog task edit <id> --plan "Implementation plan"
backlog task edit <id> --append-notes "Progress or verification note"
backlog task edit <id> --check-ac 1
backlog task edit <id> --final-summary "Summary of work and verification"
backlog board
backlog browser --no-open
```

If multi-line content is needed through the CLI, the implementation should use Backlog.md's documented shell-safe multi-line patterns instead of embedding escaped `\n` strings and assuming they will expand.

## Repo Setup

Implementation planning should initialize Backlog.md in the repo if no Backlog.md project exists.

Preferred setup:

- Project-local storage in `backlog/`.
- MCP connector as the primary agent integration.
- Root `AGENTS.md` updated manually or via Backlog.md's instruction updater, with user review if the tool would modify additional files.
- `autoCommit=false`.
- `bypassGitHooks=false`.
- Definition of Done defaults aligned to this repo's gates:
  - acceptance criteria satisfied
  - tests or verification recorded
  - docs updated when relevant
  - Bandit run for touched code when applicable, or non-code/environment skip documented
  - final summary added
  - known skips or blockers documented

If MCP client registration needs to edit user-level Codex configuration outside the repo, the implementation should request approval or provide the exact manual setup step instead of silently changing external files.

The Backlog.md instruction updater is known to update all supported agent instruction files, including `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, and `.github/copilot-instructions.md`. Because this task only asked for root `AGENTS.md`, implementation should prefer a manual `AGENTS.md` edit unless the user explicitly approves synchronizing all agent instruction files.

## AGENTS.md Placement

The new `AGENTS.md` section should live near the existing development process and planning guidance. A dedicated heading such as `### Backlog.md Task Tracking` keeps it discoverable without mixing it into code style or architecture guidance.

The section should say:

- Backlog.md is required for repo-changing work.
- Backlog.md does not replace superpowers specs/plans.
- Read the official Backlog.md workflow through MCP before task operations.
- Search first, then create only if no matching task exists.
- Keep status, notes, plan links, verification, and final summary current.
- Use CLI fallback only when MCP is unavailable.
- Creating or updating Backlog.md task records is the tracking mechanism itself and does not require a separate recursive task.
- Do not manually edit task markdown except under explicit exception.
- Commit Backlog.md task changes with the related work unless the user asks otherwise.

## Data Flow

1. User requests repo-changing work.
2. Agent reads Backlog.md official workflow instructions if needed.
3. Agent searches Backlog.md for an existing task.
4. Agent uses an existing task or creates a new task with clear description and acceptance criteria.
5. If the task needs design, the agent uses the existing superpowers brainstorming/spec flow and links the spec from the Backlog.md task.
6. If the task needs implementation planning, the agent uses the existing superpowers planning flow and links the plan from the Backlog.md task.
7. Agent moves the task through the board and appends useful notes as work proceeds.
8. Agent records verification and final summary before marking the task done.
9. Git commits include the relevant Backlog.md task changes alongside the related work.

For this initial adoption, the already-approved design spec plus the first setup commit are the bootstrap exception. After Backlog.md is initialized, the implementation should create or update the Backlog.md adoption task before making non-Backlog follow-up edits.

## Error Handling

If Backlog.md is not installed:

- Report that task tracking setup is blocked.
- Do not start repo file edits unless the user explicitly approves a temporary exception.
- Provide the official install/init path in the implementation plan.

If MCP is unavailable but the CLI works:

- Continue using the CLI fallback.
- Record in the task notes that CLI fallback was used.

If task scope expands:

- Split the work into smaller Backlog.md tasks before continuing.
- Link related tasks with dependencies or references when appropriate.

If there is already a GitHub issue, PR, spec, or plan:

- Link it from the Backlog.md task.
- Avoid copying the full artifact into task notes unless a short summary is needed for context.

## Verification

For the adoption task itself:

1. Confirm Backlog.md setup files exist after initialization.
2. Confirm root `AGENTS.md` contains a dedicated Backlog.md workflow section.
3. Confirm the instructions require Backlog.md tasks for repo file changes.
4. Confirm the instructions preserve existing superpowers and repo quality gates.
5. Confirm Backlog.md auto-commit and hook bypass are not enabled.
6. Run a read-only Backlog.md command after initialization if the CLI is available.
7. Inspect the diff and run `git diff --check`.

For future development tasks:

- Record verification commands and outcomes in the Backlog.md task.
- Record known skips or environment blockers in the task.
- Add a final summary before marking the task done.

## Resolved Implementation Decisions

1. `backlog init` should be treated as scriptable/non-interactive in this environment.
2. MCP client registration may require user-level config approval; the implementation must verify this instead of assuming it can silently complete registration.
3. Backlog.md's instruction updater touches all supported agent instruction files, so this task should update root `AGENTS.md` manually unless broader synchronization is explicitly approved.
4. This spec and the first setup commit are the bootstrap exception. Once Backlog.md exists, the adoption task should be represented in Backlog.md before non-Backlog follow-up edits continue.

## Acceptance Criteria

1. Backlog.md is initialized for the repo.
2. The repo uses Backlog.md's official MCP-first guidance.
3. Root `AGENTS.md` requires a Backlog.md task for any repo file change.
4. The policy explicitly allows read-only investigation without a task.
5. The policy explicitly preserves superpowers specs/plans and existing quality gates.
6. Backlog.md CLI fallback is documented.
7. Failure modes are documented clearly enough for agents to pause rather than silently bypass tracking.
8. The adoption diff passes markdown/diff sanity checks.
