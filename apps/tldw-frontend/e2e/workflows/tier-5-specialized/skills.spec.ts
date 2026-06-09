/**
 * Skills E2E Tests (Tier 5)
 *
 * Tests the /skills page:
 * - Page loads with connection gate or skills manager
 * - Skills list table or empty state renders
 * - Create button present when skills API available
 *
 * Run: npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts
 */
import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { seedAuth, TEST_CONFIG } from "../../utils/helpers"

const seededSkillSummary = {
  name: "summarize",
  description: "Summarize source material",
  argument_hint: "[text]",
  user_invocable: true,
  disable_model_invocation: false,
  context: "inline",
}

const seededSkillResponse = {
  ...seededSkillSummary,
  id: "skill-summarize",
  allowed_tools: null,
  model: null,
  content:
    "---\ndescription: Summarize source material\nargument-hint: \"[text]\"\ncontext: inline\n---\n\nSummarize this source: $ARGUMENTS",
  raw_content: null,
  supporting_files: null,
  directory_path: "/mock/skills/summarize",
  created_at: "2026-06-01T00:00:00Z",
  last_modified: "2026-06-01T00:00:00Z",
  version: 1,
}

type TldwConnectionStore = {
  getState: () => { state: Record<string, unknown> }
  setState: (value: { state: Record<string, unknown> }) => void
}

type TldwConnectionStoreWindow = Window & {
  __tldw_useConnectionStore?: TldwConnectionStore
}

const fulfillJson = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

async function mockSkillsBeginnerApi(page: Page) {
  let seeded = false

  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      openapi: "3.0.0",
      info: { title: "Mock tldw API", version: "test" },
      paths: {
        "/api/v1/skills": { get: {}, post: {} },
        "/api/v1/skills/context": { get: {} },
        "/api/v1/skills/seed": { post: {} },
        "/api/v1/skills/{name}": { get: {}, put: {}, delete: {} },
        "/api/v1/skills/{name}/execute": { post: {} },
      },
    })
  })

  await page.route(/\/api\/v1\/skills(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() !== "GET") {
      await fulfillJson(route, {}, 405)
      return
    }
    const skills = seeded ? [seededSkillSummary] : []
    await fulfillJson(route, {
      skills,
      count: skills.length,
      total: skills.length,
      limit: 10,
      offset: 0,
    })
  })

  await page.route(/\/api\/v1\/skills\/context(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      available_skills: seeded ? [seededSkillSummary] : [],
      context_text: seeded ? "/skill summarize [text]" : "",
    })
  })

  await page.route(/\/api\/v1\/skills\/seed(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() !== "POST") {
      await fulfillJson(route, {}, 405)
      return
    }
    seeded = true
    await fulfillJson(route, { seeded: ["summarize"], count: 1 })
  })

  await page.route(/\/api\/v1\/skills\/summarize(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, seededSkillResponse)
  })

  await page.route(
    /\/api\/v1\/skills\/summarize\/execute(?:\/)?(?:\?.*)?$/,
    async (route) => {
      if (route.request().method() !== "POST") {
        await fulfillJson(route, {}, 405)
        return
      }
      await fulfillJson(route, {
        skill_name: "summarize",
        rendered_prompt: "Summarize this source: A long article about Skills UX",
        allowed_tools: null,
        model_override: null,
        execution_mode: "inline",
        fork_output: null,
      })
    }
  )
}

async function forceSkillsConnectionState(page: Page) {
  await page.waitForFunction(
    () =>
      typeof (window as TldwConnectionStoreWindow).__tldw_useConnectionStore
        ?.getState === "function",
    null,
    { timeout: 15_000 }
  )
  await page.evaluate(() => {
    const store = (window as TldwConnectionStoreWindow).__tldw_useConnectionStore
    if (!store) throw new Error("Connection store is unavailable")
    const prev = store.getState().state
    const now = Date.now()
    store.setState({
      state: {
        ...prev,
        phase: "connected",
        isConnected: true,
        isChecking: false,
        offlineBypass: true,
        errorKind: "none",
        lastError: null,
        lastStatusCode: null,
        lastCheckedAt: now,
        knowledgeStatus: "ready",
        knowledgeLastCheckedAt: now,
        knowledgeError: null,
        configStep: "health",
        hasCompletedFirstRun: true,
      },
    })
  })
}

test.describe("Skills beginner journey (mocked)", () => {
  test("seeds built-ins, opens test run, and persists after refresh", async ({
    page,
    diagnostics,
  }) => {
    await mockSkillsBeginnerApi(page)
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)

    await expect(
      page.getByRole("heading", { name: "Start with a reusable skill" })
    ).toBeVisible()
    const emptyState = page.getByTestId("skills-empty-state")
    await expect(emptyState).toBeVisible()

    await emptyState.getByRole("button", { name: "Seed built-ins" }).click()

    const successActions = page.getByTestId("skills-success-actions")
    await expect(successActions).toContainText("Built-in skills seeded")
    await expect(successActions.getByRole("button", { name: "Test summarize" }))
      .toBeVisible()
    await expect(page.getByText("Summarize source material")).toBeVisible()

    await successActions.getByRole("button", { name: "Test summarize" }).click()
    const previewDialog = page.getByRole("dialog", { name: "Preview Skill" })
    await expect(previewDialog).toBeVisible()
    await previewDialog
      .getByPlaceholder("Enter test arguments...")
      .fill("A long article about Skills UX")
    await previewDialog.getByRole("button", { name: "Preview" }).click()
    await expect(previewDialog.getByText("Rendered Prompt")).toBeVisible()
    await expect(
      previewDialog.locator("textarea").last()
    ).toHaveValue("Summarize this source: A long article about Skills UX")

    await page.reload({ waitUntil: "domcontentloaded" })
    await forceSkillsConnectionState(page)
    await expect(page.getByText("Summarize source material")).toBeVisible()
    await expect(page.getByTestId("skills-empty-state")).toHaveCount(0)

    await assertNoCriticalErrors(diagnostics)
  })
})

test.describe("Skills", () => {
  test.beforeEach(async ({ authedPage, serverInfo }) => {
    void authedPage
    skipIfServerUnavailable(serverInfo)
  })

  test("page loads with interactive elements", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/skills", { waitUntil: "domcontentloaded" })

    const unsupportedHeading = authedPage.getByRole("heading", {
      name: /skills (are )?not available/i,
    })
    const skillsTable = authedPage.locator("table")
    const createButtons = authedPage.getByRole("button", {
      name: /new skill|add skill|create/i,
    })

    await expect(
      unsupportedHeading.or(skillsTable).or(createButtons).first()
    ).toBeVisible({ timeout: 15_000 })

    await assertNoCriticalErrors(diagnostics)
  })

  test("skills list fires API on load when available", async ({
    authedPage,
    diagnostics,
  }) => {
    let skillsRequestMade = false
    const handler = (req: import("@playwright/test").Request) => {
      if (req.url().includes("/api/v1/skills") && req.method() === "GET") {
        skillsRequestMade = true
      }
    }
    authedPage.on("request", handler)

    await authedPage.goto("/skills", { waitUntil: "domcontentloaded" })

    const unsupportedHeading = authedPage.getByRole("heading", {
      name: /skills (are )?not available/i,
    })
    const skillsTable = authedPage.locator("table")
    const createButtons = authedPage.getByRole("button", {
      name: /new skill|add skill|create/i,
    })

    await expect(
      unsupportedHeading.or(skillsTable).or(createButtons).first()
    ).toBeVisible({ timeout: 15_000 })

    const unsupportedVisible = await unsupportedHeading
      .isVisible()
      .catch(() => false)

    if (unsupportedVisible) {
      await expect(unsupportedHeading).toBeVisible()
      expect(skillsRequestMade).toBe(false)
    } else {
      await expect
        .poll(() => skillsRequestMade, { timeout: 15_000 })
        .toBe(true)
    }

    authedPage.removeListener("request", handler)

    await assertNoCriticalErrors(diagnostics)
  })

  test("page has buttons or table for skill management", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/skills", { waitUntil: "domcontentloaded" })

    const unsupportedHeading = authedPage.getByRole("heading", {
      name: /skills (are )?not available/i,
    })
    const unsupportedVisible = await unsupportedHeading
      .isVisible({ timeout: 5_000 })
      .catch(() => false)

    if (unsupportedVisible) {
      await expect(
        authedPage.getByText(/skills api/i)
      ).toBeVisible({ timeout: 10_000 })
    } else {
      const interactiveElements = authedPage.locator(
        "button, input, select, textarea, table"
      )
      await expect(interactiveElements.first()).toBeVisible({ timeout: 15_000 })
      expect(await interactiveElements.count()).toBeGreaterThan(0)
    }

    await assertNoCriticalErrors(diagnostics)
  })
})
