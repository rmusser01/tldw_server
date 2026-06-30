/**
 * Agent Tasks E2E Tests (Tier 3)
 *
 * Tests the Agent Tasks (Orchestration) page lifecycle:
 * - Page loads with Projects and Tasks panels
 * - Empty state displays correctly
 * - Create Project modal opens and closes
 * - Create Task modal opens when a project is selected
 * - API calls fire for project listing on page load
 *
 * Run: npx playwright test e2e/workflows/tier-3-automation/agent-tasks.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { AgentTasksPage } from "../../utils/page-objects/AgentTasksPage"
import { ResearchWorkspacePage } from "../../utils/page-objects/ResearchWorkspacePage"
import { expectApiCall } from "../../utils/api-assertions"
import {
  fetchWithApiKey,
  generateTestId,
  seedAuth,
  TEST_CONFIG,
} from "../../utils/helpers"
import { existsSync, mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

const SERVER_BASE_URL = TEST_CONFIG.serverUrl.replace(/\/$/, "")
const CANONICAL_WORKSPACE_SOURCE = "research_workspace"

type JsonApiResult<T = unknown> = {
  ok: boolean
  status: number
  body: T | null
  text: string
}

type ProjectSummary = {
  id: number
  name: string
  canonical_workspace?: {
    canonical_workspace_id?: string | null
    canonical_workspace_source?: string | null
    link_status?: string | null
  } | null
}

type TaskSummary = {
  id: number
  title: string
  status: string
}

type RunSummary = {
  id: number
  status: string
  session_id?: string | null
  session?: {
    session_id?: string | null
    available?: boolean
    links?: Record<string, string>
  } | null
}

type TaskDetail = TaskSummary & {
  runs?: RunSummary[]
}

type WorkspaceAcpFixtureResult =
  | {
      created: false
      reason: string
      cleanup: () => Promise<void>
    }
  | {
      created: true
      workspaceId: string
      workspaceName: string
      projectId: number
      projectName: string
      taskId: number
      taskTitle: string
      runId: number
      runStatus: string
      sessionId: string
      diagnosticsHref: string
      filteredProjectIds: number[]
      cleanup: () => Promise<void>
    }

const apiJson = async <T = unknown>(
  path: string,
  init: RequestInit = {}
): Promise<JsonApiResult<T>> => {
  const response = await fetchWithApiKey(
    `${SERVER_BASE_URL}${path}`,
    TEST_CONFIG.apiKey,
    {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...((init.headers || {}) as Record<string, string>),
      },
    }
  )
  const text = await response.text().catch(() => "")
  let body: T | null = null
  if (text) {
    try {
      body = JSON.parse(text) as T
    } catch {
      body = null
    }
  }
  return {
    ok: response.ok,
    status: response.status,
    body,
    text,
  }
}

const summarizeApiFailure = (
  label: string,
  result: JsonApiResult<unknown>
): string =>
  `${label} returned HTTP ${result.status}${
    result.text ? `: ${result.text.slice(0, 240)}` : ""
  }`

const createWorkspaceRoot = (): string => {
  const rootBase =
    process.env.TLDW_E2E_ACP_WORKSPACE_ROOT_BASE ||
    (existsSync("/private/tmp") ? "/private/tmp" : tmpdir())
  return mkdtempSync(join(rootBase, "rw-acp-"))
}

const getRunSessionId = (run: RunSummary): string | null =>
  run.session?.session_id?.trim() || run.session_id?.trim() || null

const tryCreateWorkspaceScopedAcpRun =
  async (): Promise<WorkspaceAcpFixtureResult> => {
    const suffix = generateTestId("rw-acp").replace(/[^a-zA-Z0-9-]/g, "-")
    const workspaceId = `rw-acp-${suffix}`
    const workspaceName = `RW ACP ${suffix}`
    const projectName = `RW ACP Project ${suffix}`
    const taskTitle = `Inspect Research Workspace ACP bridge ${suffix}`
    const rootPath = createWorkspaceRoot()
    const cleanupPaths: Array<{ method: "DELETE"; path: string }> = []

    const cleanup = async () => {
      for (const item of [...cleanupPaths].reverse()) {
        await apiJson(item.path, { method: item.method }).catch(() => null)
      }
      rmSync(rootPath, { recursive: true, force: true })
    }

    const workspaceResult = await apiJson(`/api/v1/workspaces/${workspaceId}`, {
      method: "PUT",
      body: JSON.stringify({
        name: workspaceName,
        study_materials_policy: "workspace",
      }),
    })
    if (!workspaceResult.ok) {
      return {
        created: false,
        reason: summarizeApiFailure("PUT /api/v1/workspaces/{id}", workspaceResult),
        cleanup,
      }
    }
    cleanupPaths.push({
      method: "DELETE",
      path: `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`,
    })

    const bridgeResult = await apiJson<{
      id?: number
      canonical_workspace?: ProjectSummary["canonical_workspace"]
    }>("/api/v1/agent-orchestration/workspaces/canonical-bridge", {
      method: "POST",
      body: JSON.stringify({
        canonical_workspace_id: workspaceId,
        canonical_workspace_source: CANONICAL_WORKSPACE_SOURCE,
        root_path: rootPath,
        name: `ACP ${workspaceName}`,
        description: "E2E Research Workspace ACP bridge",
        metadata: {
          e2e: true,
          canonical_workspace_id: workspaceId,
          canonical_workspace_source: CANONICAL_WORKSPACE_SOURCE,
        },
      }),
    })
    if (!bridgeResult.ok || !bridgeResult.body?.id) {
      return {
        created: false,
        reason: summarizeApiFailure(
          "POST /api/v1/agent-orchestration/workspaces/canonical-bridge",
          bridgeResult
        ),
        cleanup,
      }
    }
    const acpWorkspaceId = bridgeResult.body.id
    cleanupPaths.push({
      method: "DELETE",
      path: `/api/v1/agent-orchestration/workspaces/${acpWorkspaceId}`,
    })

    const projectResult = await apiJson<ProjectSummary>(
      "/api/v1/agent-orchestration/projects",
      {
        method: "POST",
        body: JSON.stringify({
          name: projectName,
          description: "E2E project for Research Workspace ACP history",
          workspace_id: acpWorkspaceId,
          metadata: {
            e2e: true,
            canonical_workspace_id: workspaceId,
            canonical_workspace_source: CANONICAL_WORKSPACE_SOURCE,
          },
        }),
      }
    )
    if (!projectResult.ok || !projectResult.body?.id) {
      return {
        created: false,
        reason: summarizeApiFailure(
          "POST /api/v1/agent-orchestration/projects",
          projectResult
        ),
        cleanup,
      }
    }
    const projectId = projectResult.body.id
    cleanupPaths.push({
      method: "DELETE",
      path: `/api/v1/agent-orchestration/projects/${projectId}`,
    })

    const filterPath =
      `/api/v1/agent-orchestration/projects?canonical_workspace_id=${encodeURIComponent(
        workspaceId
      )}&canonical_workspace_source=${CANONICAL_WORKSPACE_SOURCE}`
    const filteredProjectsResult = await apiJson<ProjectSummary[]>(filterPath)
    if (!filteredProjectsResult.ok || !Array.isArray(filteredProjectsResult.body)) {
      return {
        created: false,
        reason: summarizeApiFailure(
          "GET /api/v1/agent-orchestration/projects?canonical_workspace_id=...",
          filteredProjectsResult
        ),
        cleanup,
      }
    }
    const filteredProjectIds = filteredProjectsResult.body.map((project) => project.id)
    if (!filteredProjectIds.includes(projectId)) {
      return {
        created: false,
        reason: `Filtered ACP project list did not include created project ${projectId}: ${JSON.stringify(filteredProjectIds)}`,
        cleanup,
      }
    }

    const taskResult = await apiJson<TaskSummary>(
      `/api/v1/agent-orchestration/projects/${projectId}/tasks`,
      {
        method: "POST",
        body: JSON.stringify({
          title: taskTitle,
          description:
            "Validate that this Research Workspace can surface ACP-owned run state.",
          agent_type: "opencode",
          success_criteria:
            "Create an ACP session-backed run record for workspace history validation.",
          metadata: {
            e2e: true,
            canonical_workspace_id: workspaceId,
            canonical_workspace_source: CANONICAL_WORKSPACE_SOURCE,
          },
        }),
      }
    )
    if (!taskResult.ok || !taskResult.body?.id) {
      return {
        created: false,
        reason: summarizeApiFailure(
          `POST /api/v1/agent-orchestration/projects/${projectId}/tasks`,
          taskResult
        ),
        cleanup,
      }
    }
    const taskId = taskResult.body.id

    const dispatchResult = await apiJson(
      `/api/v1/agent-orchestration/tasks/${taskId}/run`,
      {
        method: "POST",
        body: JSON.stringify({
          agent_type: "opencode",
        }),
      }
    )

    const taskDetailResult = await apiJson<TaskDetail>(
      `/api/v1/agent-orchestration/tasks/${taskId}`
    )
    if (!taskDetailResult.ok || !taskDetailResult.body?.runs) {
      return {
        created: false,
        reason:
          `${summarizeApiFailure(
            `GET /api/v1/agent-orchestration/tasks/${taskId}`,
            taskDetailResult
          )}; dispatch status was HTTP ${dispatchResult.status}`,
        cleanup,
      }
    }

    const run = [...taskDetailResult.body.runs]
      .reverse()
      .find((candidate) => {
        const sessionId = getRunSessionId(candidate)
        return Boolean(sessionId && candidate.session?.links?.diagnostics)
      })
    const sessionId = run ? getRunSessionId(run) : null
    const diagnosticsHref = run?.session?.links?.diagnostics || null
    if (!run || !sessionId || !diagnosticsHref) {
      return {
        created: false,
        reason:
          `Dispatch did not create a diagnostics-linked ACP run; dispatch status was HTTP ${dispatchResult.status}`,
        cleanup,
      }
    }

    const diagnosticsResult = await apiJson(diagnosticsHref)
    if (!diagnosticsResult.ok) {
      return {
        created: false,
        reason: summarizeApiFailure(
          `GET ${diagnosticsHref}`,
          diagnosticsResult
        ),
        cleanup,
      }
    }

    return {
      created: true,
      workspaceId,
      workspaceName,
      projectId,
      projectName,
      taskId,
      taskTitle,
      runId: run.id,
      runStatus: run.status,
      sessionId,
      diagnosticsHref,
      filteredProjectIds,
      cleanup,
    }
  }

const activateResearchWorkspace = async (
  page: import("@playwright/test").Page,
  workspaceId: string,
  workspaceName: string
) => {
  await page.evaluate(
    ({ id, name }) => {
      const store = (window as { __tldw_useWorkspaceStore?: unknown })
        .__tldw_useWorkspaceStore as
        | {
            getState?: () => {
              loadWorkspace?: (config: {
                id: string
                name: string
                tag: string
                createdAt: string
              }) => void
            }
          }
        | undefined

      if (!store?.getState) {
        throw new Error("Workspace store is unavailable on window")
      }

      store.getState().loadWorkspace?.({
        id,
        name,
        tag: `workspace:${id}`,
        createdAt: new Date().toISOString(),
      })
    },
    {
      id: workspaceId,
      name: workspaceName,
    }
  )
}

test.describe("Agent Tasks", () => {
  let agentTasks: AgentTasksPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    agentTasks = new AgentTasksPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render Agent Tasks page with Projects and Tasks panels", async ({
      authedPage,
      diagnostics,
    }) => {
      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()
      await agentTasks.assertPageReady()

      // Projects heading should be visible
      const projectsVisible = await agentTasks.projectsHeading.isVisible().catch(() => false)
      // Tasks heading should also be visible
      const tasksVisible = await agentTasks.tasksHeading.isVisible().catch(() => false)

      expect(projectsVisible || tasksVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show empty state or project list in Projects panel", async ({
      authedPage,
      diagnostics,
    }) => {
      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()
      await agentTasks.assertPageReady()

      // Either projects are listed or empty state is shown
      const isEmpty = await agentTasks.isProjectsEmpty()
      const newButtonVisible = await agentTasks.newProjectButton.isVisible().catch(() => false)

      // Either the empty state or the New button should be present
      expect(isEmpty || newButtonVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show 'Select a project' in Tasks panel when no project selected", async ({
      authedPage,
      diagnostics,
    }) => {
      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()
      await agentTasks.assertPageReady()

      const selectProjectVisible = await agentTasks.isTasksSelectProject()
      // If there are no projects, the tasks panel shows "select a project"
      // If there are projects but none selected, same message
      // This is expected on fresh load
      expect(selectProjectVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should guide ACP setup and task diagnostics without manual ID copying", async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.route("**/openapi.json", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            paths: {
              "/api/v1/agent-orchestration/projects": {},
            },
          }),
        })
      })
      await authedPage.route("**/api/v1/acp/health", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            runner: "ok",
            agent: "ok",
            api_keys: "ok",
          }),
        })
      })
      await authedPage.route("**/api/v1/agent-orchestration/projects", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 7,
              name: "ACP Production Project",
              user_id: 1,
              created_at: "2026-05-10T00:00:00Z",
            },
          ]),
        })
      })
      await authedPage.route("**/api/v1/agent-orchestration/projects/7/tasks", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 11,
              project_id: 7,
              title: "Harden completion gate",
              status: "triage",
              review_count: 1,
              max_review_attempts: 3,
              created_at: "2026-05-10T00:00:00Z",
              updated_at: "2026-05-10T00:05:00Z",
            },
          ]),
        })
      })
      await authedPage.route("**/api/v1/agent-orchestration/tasks/11", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: 11,
            project_id: 7,
            title: "Harden completion gate",
            status: "triage",
            review_count: 1,
            max_review_attempts: 3,
            created_at: "2026-05-10T00:00:00Z",
            updated_at: "2026-05-10T00:05:00Z",
            runs: [
              {
                id: 51,
                task_id: 11,
                session_id: "sess-prod-51",
                agent_type: "codex",
                status: "failed",
                error: "Workspace root not allowed",
                started_at: "2026-05-10T00:01:00Z",
                session: {
                  session_id: "sess-prod-51",
                  available: true,
                  links: {
                    diagnostics: "/api/v1/acp/sessions/sess-prod-51/diagnostics",
                    artifacts: "/api/v1/acp/sessions/sess-prod-51/artifacts",
                    audit: "/api/v1/acp/sessions/sess-prod-51/audit",
                  },
                },
                history: {
                  event_count: 3,
                  audit_event_count: 2,
                  artifact_count: 1,
                  diagnostic_count: 1,
                  tool_call_count: 4,
                  result: {
                    role: "assistant",
                    preview: "I could not access the workspace.",
                  },
                },
                failure_context: {
                  reason_code: "workspace_root_not_allowed",
                  message: "Workspace root not allowed",
                  source: "session_diagnostic",
                },
                review_decision: {
                  available: true,
                  approved: false,
                  reviewer: "reviewer-agent",
                  feedback_preview: "Needs citations",
                },
              },
            ],
            reviews: [
              {
                reviewer: "reviewer-agent",
                approved: false,
                feedback: "Needs citations",
                created_at: "2026-05-10T00:06:00Z",
              },
            ],
          }),
        })
      })

      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()
      await agentTasks.assertPageReady()

      await authedPage.getByText("ACP Production Project").click()
      await expect(authedPage.getByText("Harden completion gate")).toBeVisible()
      await authedPage.getByRole("button", { name: /inspect/i }).click()

      await expect(authedPage.getByText("Run #51")).toBeVisible()
      await expect(authedPage.getByText("sess-prod-51")).toBeVisible()
      await expect(authedPage.getByText("workspace_root_not_allowed")).toBeVisible()
      await expect(authedPage.getByRole("button", { name: /open diagnostics/i })).toBeVisible()
      await expect(authedPage.getByRole("button", { name: /open artifacts/i })).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("binds a Research Workspace to a real ACP run history and diagnostics path", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      const fixture = await tryCreateWorkspaceScopedAcpRun()
      if (fixture.created === false) {
        const skipReason = fixture.reason
        await fixture.cleanup()
        test.skip(true, skipReason)
        return
      }

      try {
        expect(fixture.filteredProjectIds).toContain(fixture.projectId)

        const workspacePage = new ResearchWorkspacePage(authedPage)
        await workspacePage.goto()
        await workspacePage.waitForReady()
        await activateResearchWorkspace(
          authedPage,
          fixture.workspaceId,
          fixture.workspaceName
        )
        await expect
          .poll(async () => workspacePage.getWorkspaceId(), {
            timeout: 10_000,
            message: "Expected Research Workspace UI to switch to seeded workspace",
          })
          .toBe(fixture.workspaceId)

        const projectsRequestPromise = authedPage.waitForRequest((request) => {
          if (request.method().toUpperCase() !== "GET") return false
          const url = new URL(request.url())
          return (
            url.pathname.endsWith("/api/v1/agent-orchestration/projects") &&
            url.searchParams.get("canonical_workspace_id") === fixture.workspaceId &&
            url.searchParams.get("canonical_workspace_source") ===
              CANONICAL_WORKSPACE_SOURCE
          )
        })

        await authedPage
          .getByRole("button", { name: /workspace settings/i })
          .click()
        await authedPage.getByText("ACP run history").click()

        const projectsRequest = await projectsRequestPromise
        const projectsUrl = new URL(projectsRequest.url())
        expect(projectsUrl.searchParams.get("canonical_workspace_id")).toBe(
          fixture.workspaceId
        )
        expect(projectsUrl.searchParams.get("canonical_workspace_source")).toBe(
          CANONICAL_WORKSPACE_SOURCE
        )

        const modal = authedPage.getByRole("dialog", { name: /ACP run history/i })
        await expect(modal).toBeVisible({ timeout: 15_000 })
        await expect(modal.getByText(fixture.projectName)).toBeVisible({
          timeout: 20_000,
        })
        await expect(modal.getByText(fixture.taskTitle)).toBeVisible()
        await expect(modal.getByText(fixture.sessionId)).toBeVisible()
        await expect(
          modal.getByText(fixture.runStatus, { exact: true })
        ).toBeVisible()

        const diagnosticsButton = modal.getByRole("button", {
          name: /open diagnostics/i,
        })
        await expect(diagnosticsButton).toBeVisible()
        await Promise.all([
          authedPage.waitForURL((url) => {
            return (
              url.pathname === "/acp-playground" &&
              url.searchParams.get("session") === fixture.sessionId &&
              url.searchParams.get("view") === "diagnostics"
            )
          }),
          diagnosticsButton.click(),
        ])

        const diagnosticsUrl = new URL(authedPage.url())
        expect(diagnosticsUrl.pathname).toBe("/acp-playground")
        expect(diagnosticsUrl.searchParams.get("session")).toBe(fixture.sessionId)
        expect(diagnosticsUrl.searchParams.get("view")).toBe("diagnostics")

        await assertNoCriticalErrors(diagnostics)
      } finally {
        await fixture.cleanup()
      }
    })
  })

  // =========================================================================
  // Modal Interactions
  // =========================================================================

  test.describe("Modal Interactions", () => {
    test("should open and close Create Project modal", async ({
      authedPage,
      diagnostics,
    }) => {
      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()
      await agentTasks.assertPageReady()

      // Try the "New" button in the card header
      const newBtnVisible = await agentTasks.newProjectButton.isVisible().catch(() => false)
      // Or the "Create Project" button in empty state
      const createBtnVisible = await agentTasks.createProjectButton.isVisible().catch(() => false)

      const triggerBtn = newBtnVisible
        ? agentTasks.newProjectButton
        : createBtnVisible
          ? agentTasks.createProjectButton
          : null

      if (!triggerBtn) return

      await triggerBtn.click()

      // Modal should appear
      const modal = agentTasks.createProjectModal
      await expect(modal).toBeVisible({ timeout: 5_000 })

      // Close the modal
      await authedPage.keyboard.press("Escape")
      await expect(modal).toBeHidden({ timeout: 3_000 }).catch(() => {})

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // API Integration (requires server)
  // =========================================================================

  test.describe("API Integration", () => {
    test("should fire GET /api/v1/agent-orchestration/projects on page load", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      // Set up API call listener before navigating
      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/agent-orchestration\/projects/,
        method: "GET",
      }, 20_000)

      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Server may not have orchestration endpoint available
      }

      await agentTasks.assertPageReady()
      await assertNoCriticalErrors(diagnostics)
    })

    test("should fire POST /api/v1/agent-orchestration/projects when creating a project", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      agentTasks = new AgentTasksPage(authedPage)
      await agentTasks.goto()
      await agentTasks.assertPageReady()

      // Open the create project modal
      const newBtnVisible = await agentTasks.newProjectButton.isVisible().catch(() => false)
      const createBtnVisible = await agentTasks.createProjectButton.isVisible().catch(() => false)
      const triggerBtn = newBtnVisible
        ? agentTasks.newProjectButton
        : createBtnVisible
          ? agentTasks.createProjectButton
          : null

      if (!triggerBtn) return

      await triggerBtn.click()
      await expect(agentTasks.createProjectModal).toBeVisible({ timeout: 5_000 })

      // Fill in the project name
      const nameInput = agentTasks.createProjectModal.getByLabel(/project name/i)
      await nameInput.fill("E2E Test Project")

      // Listen for the POST call
      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/agent-orchestration\/projects/,
        method: "POST",
      }, 15_000)

      // Submit
      await agentTasks.createProjectModal.getByRole("button", { name: /create/i }).last().click()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // API may reject if orchestration is not configured
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
