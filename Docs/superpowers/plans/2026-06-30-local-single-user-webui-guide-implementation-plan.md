# Local Single-User WebUI Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local single-user setup guide self-contained for running the local FastAPI server and the Next.js WebUI.

**Architecture:** This is a documentation-only change. Keep the existing local API flow intact, extend the current WebUI startup flow in the `## Start` section, and finish by updating verification, troubleshooting, optional add-ons, and Backlog task state. No server, frontend, Makefile, Docker, or environment-template behavior changes are part of this plan.

**Tech Stack:** Markdown documentation, Backlog.md MCP task tracking, shell-based docs hygiene checks.

---

## File Structure

- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`
  - Responsibility: canonical local single-user guide for API and WebUI setup.
- Modify: `backlog/tasks/task-12075 - Update-Local-Single-User-Setup-guide-with-WebUI.md`
  - Responsibility: task status, implementation notes, verification, and final summary.
- Existing reference: `Docs/superpowers/specs/2026-06-30-local-single-user-webui-guide-design.md`
  - Responsibility: approved design constraints.
- Existing reference: `README.md`
  - Responsibility: existing WebUI add-on and advanced networking wording.
- Existing reference: `apps/tldw-frontend/README.md`
  - Responsibility: deeper WebUI development and auth-mode details.
- Existing reference: `apps/DEVELOPMENT.md`
  - Responsibility: extension/WebUI development workflow.

## Task 1: Update Local Guide Happy Path

**Files:**
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`

- [x] **Step 1: Read the current local guide and approved design**

Run:

```bash
sed -n '1,180p' Docs/Getting_Started/Profile_Local_Single_User.md
sed -n '1,220p' Docs/superpowers/specs/2026-06-30-local-single-user-webui-guide-design.md
```

Expected: the guide still has the local API setup, and the spec says the guide should become self-contained for the WebUI happy path.

- [x] **Step 2: Add WebUI prerequisites to `Prepare`**

In `Docs/Getting_Started/Profile_Local_Single_User.md`, update the prerequisites list to:

```markdown
Prerequisites:

- Python 3.10+
- `ffmpeg`
- Git
- Bun for the WebUI (`npm` also works as a fallback)
```

Expected: the guide still lists the original API prerequisites and now names Bun for the WebUI.

- [x] **Step 3: Extend the WebUI startup path in `Start`**

In `Docs/Getting_Started/Profile_Local_Single_User.md`, extend the existing WebUI startup block under `## Start` so it contains:

````markdown
```bash
cd apps/tldw-frontend
cp .env.local.example .env.local
```

Edit `.env.local` so it points the WebUI at the local API:

```dotenv
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_VERSION=v1
# Optional in single-user mode:
# NEXT_PUBLIC_X_API_KEY=your_single_user_api_key
```

`NEXT_PUBLIC_X_API_KEY` is browser-visible. Use it only for local single-user convenience, and avoid exposing this setup directly to the public internet.

Install and start the WebUI:

```bash
bun install
bun run dev -- -p 8080

# npm fallback:
# npm install
# npm run dev -- -p 8080
```

Open http://localhost:8080.
````

Expected: the local guide contains one complete WebUI setup path and no longer requires the README add-on section for the happy path.

- [x] **Step 4: Add WebUI verification**

In the `Verify` manual spot checks, include a WebUI spot check:

````markdown
```bash
curl -sS http://127.0.0.1:8080 > /dev/null && echo "webui-ok"
```
````

Expected: the guide verifies both services: API at `8000`, WebUI at `8080`.

- [x] **Step 5: Update `First Value` for API or WebUI entry**

Update the first-value copy so it recognizes the WebUI setup flow and preserves the provider-independent terminal check:

```markdown
Open http://127.0.0.1:8080 to open the WebUI and complete first-time setup there. The setup completion gate is the first successful chat response from your selected hosted API key or local OpenAI-compatible provider. Immediately after that, add your first source so chat can use your own material.

If you prefer terminal verification, the CLI verify command still runs a provider-independent first-value ingest/search check. It posts a small Markdown document to `/api/v1/media/add`, then searches for `tldw-onboarding-verification-unique` through `/api/v1/media/search`.
```

Expected: first-value guidance recognizes the WebUI as a valid local starting point while preserving the existing provider-independent verification command.

## Task 2: Update Troubleshooting and References

**Files:**
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`

- [x] **Step 1: Add WebUI troubleshooting bullets**

In the `Troubleshoot` section, add these bullets after the existing port `8000` bullet:

```markdown
- If the WebUI port `8080` is in use, stop the conflicting process or start Next.js on another port with `bun run dev -- -p 8081`.
- If `bun install` fails, verify Bun with `bun --version`, or use the npm fallback commands from the WebUI section.
- If the WebUI loads but cannot reach the API, confirm `.env.local` uses the same host and port as the running API, usually `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`.
```

Expected: common WebUI local failures have direct fixes.

- [x] **Step 2: Replace optional add-on WebUI link**

Replace the current optional add-ons list with:

```markdown
- Keep provider setup in the WebUI first-run wizard for the normal path; add provider API keys to `tldw_Server_API/Config_Files/.env` only for recovery, automation, or advanced deployments, then restart the server.
- For deeper WebUI development details, see [Extension & Web UI Development Guide](../../apps/DEVELOPMENT.md) and [tldw-frontend README](../../apps/tldw-frontend/README.md).
- For LAN, mobile, reverse-proxy, or custom-host browser access, see [Run the Web UI (WIP)](../../README.md#run-the-web-ui-wip).
- Install development extras with `source .venv/bin/activate && pip install -e ".[dev]"`.
```

Expected: the old "Add the WebUI" link is gone because the guide now contains that path.

- [x] **Step 3: Check for stale README-only wording**

Run this against the guide:

```bash
rg -n "Add the WebUI|Local Profile: Add the WebUI|does not include the WebUI by default" Docs/Getting_Started/Profile_Local_Single_User.md
```

Expected: no matches.

## Task 3: Verify Documentation and Finalize Task

**Files:**
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Modify: `backlog/tasks/task-12075 - Update-Local-Single-User-Setup-guide-with-WebUI.md`

- [x] **Step 1: Review the rendered command order in plain text**

Run:

```bash
sed -n '1,220p' Docs/Getting_Started/Profile_Local_Single_User.md
```

Expected: the order is Prepare, Start, Verify, First Value, Audio Path, Troubleshoot, Optional Add-ons.

- [x] **Step 2: Check Markdown whitespace**

Run:

```bash
git diff --check -- Docs/Getting_Started/Profile_Local_Single_User.md
```

Expected: no output and exit code `0`.

- [x] **Step 3: Check referenced files exist**

Run:

```bash
test -f apps/DEVELOPMENT.md
test -f apps/tldw-frontend/README.md
test -f README.md
```

Expected: each command exits `0`.

- [x] **Step 4: Check README anchor exists**

Run:

```bash
rg -n "^### Run the Web UI \\(WIP\\)" README.md
```

Expected: one match for the README section linked from Optional Add-ons.

- [x] **Step 5: Run targeted onboarding docs hygiene if available**

Run:

```bash
if test -f Helper_Scripts/docs/check_onboarding_command_boundaries.py; then
  source .venv/bin/activate && python Helper_Scripts/docs/check_onboarding_command_boundaries.py
else
  echo "check_onboarding_command_boundaries.py not present; skipped"
fi
```

Expected: either the checker passes, or the skip message prints because the checker is absent. If the virtual environment is unavailable, record that verification blocker in `TASK-12075` and continue with the static checks above.

- [x] **Step 6: Record Bandit skip**

Use Backlog MCP `task_edit` on `TASK-12075` and add implementation notes:

```text
Bandit skipped: documentation-only Markdown changes; no Python code touched.
```

Expected: Backlog task notes explain why Bandit does not apply.

- [x] **Step 7: Update Backlog final summary and acceptance criteria**

Use Backlog MCP `task_edit` on `TASK-12075` to record:

```text
Final summary: Updated the local single-user guide to include self-contained WebUI prerequisites, env setup, start commands, verification, troubleshooting, and deeper-reference links.
Verification: Markdown whitespace check passed; referenced docs exist; README WebUI anchor checked; onboarding docs checker result recorded.
```

Expected: `TASK-12075` contains verification and final summary notes. Mark acceptance criteria and Definition of Done complete only after the guide edit and checks have actually passed.

- [x] **Step 8: Commit the documentation update**

Run:

```bash
git add Docs/Getting_Started/Profile_Local_Single_User.md "backlog/tasks/task-12075 - Update-Local-Single-User-Setup-guide-with-WebUI.md"
git commit -m "docs: add local webui setup guide"
```

Expected: commit succeeds and includes only the guide and Backlog task update.
