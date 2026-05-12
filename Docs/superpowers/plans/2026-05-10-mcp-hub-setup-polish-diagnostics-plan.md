# MCP Hub Setup Polish And Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship PR 2 of the MCP Hub walkthrough remediation by making no-auth setup states intentional, improving Tool Catalog recovery guidance, exposing compact deployment diagnostics, and documenting an isolated toy-MCP walkthrough path.

**Architecture:** Keep PR 2 frontend/docs-heavy and build on the merged PR 1 runtime refresh contract. Add small pure helpers beside MCP Hub components for setup-state classification and diagnostics derivation, then render those helpers in existing Setup workflow tabs without creating a new backend contract. Keep the E2E smoke skip-safe against unavailable live servers and keep local walkthrough isolation in documentation plus verification notes.

**Tech Stack:** React, Ant Design, TanStack Query, Vitest, Testing Library, Playwright, Markdown docs.

---

## File Map

- Modify `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`: render no-auth stdio state, hide legacy fallback unless server-level secret flow is relevant, and replace misleading list tags.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx`: add no-auth and legacy fallback visibility regressions.
- Modify `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx`: add empty-state classification, Add Server route action, Refresh discovery recovery action, and permission guidance when executable tools are absent.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx`: cover setup absence, runtime absence, refresh failure, and permission-absence guidance.
- Create `apps/packages/ui/src/components/Option/MCPHub/DeploymentDiagnosticsPanel.tsx`: compact diagnostics surface for deployment mode, request mode, API origin, health URL, and last MCP health status.
- Create `apps/packages/ui/src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx`: focused rendering coverage for quickstart and advanced mode values.
- Modify `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`: render diagnostics in the Setup workflow only.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`: assert the diagnostics panel is available without disturbing workflow routing.
- Modify `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`: add a skip-safe toy MCP smoke skeleton that verifies the UI path and records server availability constraints.
- Modify `Docs/MCP/mcp_hub_management.md`: document local toy MCP walkthrough isolation, temp-path environment, and which databases remain repo-local unless separately configured.
- Modify `backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md`: record implementation plan, verification, and final summary.

## Stage 1: No-Auth And Legacy Secret Setup States

**Goal:** Make local stdio servers without credential requirements read as complete rather than broken.

**Success Criteria:**
- No-auth stdio managed servers show `No credentials required`.
- No-auth stdio servers do not show `No auth template`, `no secret`, or `Legacy Secret Fallback`.
- Legacy fallback appears only when the active managed server has no credential slots and server-level auth is relevant or already configured.

**Tests:**
- Extend `ExternalServersTab.test.tsx` with a no-auth stdio managed server fixture.
- Add a regression that asserts `No credentials required` is visible and missing-auth copy is absent for that server.
- Add a server-level secret fixture and assert `Legacy Secret Fallback` remains visible there.

**Status:** Complete

## Stage 2: Tool Catalog Empty And Recovery Guidance

**Goal:** Make the Tool Catalog empty state explain whether the user needs setup, runtime refresh, or access/policy work.

**Success Criteria:**
- No managed server plus no tools shows Add server as the primary action.
- Managed servers plus no registry tools shows Refresh discovery as the primary action.
- Refresh errors preserve server-specific error text.
- Tools registered but no executable chat tools shows Access/policy guidance instead of implying setup is incomplete.

**Tests:**
- Extend `ToolCatalogsTab.test.tsx` for each empty-state branch.
- Mock executable-tool availability through `useMcpTools` or a small prop seam, not by inferring caller permissions from registry metadata alone.

**Status:** Complete

## Stage 3: Compact Deployment Diagnostics

**Goal:** Expose enough networking/deployment context to diagnose quickstart versus advanced API-origin mismatch without bloating MCP Hub.

**Success Criteria:**
- Setup workflow renders a compact diagnostics panel.
- Quickstart mode reports same-origin/proxied calls, relative API origin, and `/api/v1/health`.
- Advanced mode reports direct API origin, health URL, and current page origin.
- Last MCP health state/status is shown when available.

**Tests:**
- Add `DeploymentDiagnosticsPanel.test.tsx` for quickstart and advanced mode.
- Add a `McpHubPage.test.tsx` assertion that the diagnostics panel appears in Setup workflow.

**Status:** Complete

## Stage 4: Walkthrough Isolation Documentation And Smoke

**Goal:** Make the toy MCP walkthrough reproducible without accidental repo DB churn where practical.

**Success Criteria:**
- `Docs/MCP/mcp_hub_management.md` includes a local toy MCP isolation recipe using temporary paths.
- The docs explicitly call out that `DATABASE_URL` covers AuthNZ while other runtime databases need `USER_DB_BASE_DIR`, `MCP_DATABASE_URL`, and related temp paths to avoid repo-local writes.
- The E2E smoke has a skip-safe toy MCP path or records the live-server requirement when unavailable.
- Verification notes include `git status --short` cleanliness.

**Tests:**
- Add/extend Playwright MCP Hub smoke to verify the UI path and skip when no server is available.
- Run the focused smoke command if a server is already available; otherwise record the skip reason.

**Status:** Complete

## Stage 5: Verification And PR Packaging

**Goal:** Verify the focused UI/docs slice and prepare PR notes.

**Verification Commands:**
- `cd apps/packages/ui && bun run test src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --reporter=line`
- `git diff --check`
- `git status --short`

**Bandit:** Not applicable if PR 2 remains TypeScript/docs/E2E-only. If backend Python is touched, run Bandit on touched backend production files before finalizing.

**Status:** Complete
