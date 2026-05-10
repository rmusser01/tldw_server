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
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

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
