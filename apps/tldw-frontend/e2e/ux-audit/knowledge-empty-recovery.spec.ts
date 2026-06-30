import { test, expect, type Page, type Route } from "@playwright/test"

import { seedAuth, TEST_CONFIG } from "../utils/helpers"

const fulfillJson = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

async function mockKnowledgeRecoveryApi(
  page: Page,
  options: { hasWebSearch?: boolean } = {}
) {
  const hasWebSearch = options.hasWebSearch ?? true

  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      openapi: "3.0.0",
      paths: {
        "/api/v1/rag/search": { post: {} },
        "/api/v1/rag/health": { get: {} },
        ...(hasWebSearch
          ? { "/api/v1/research/websearch": { post: {} } }
          : {}),
      },
    })
  })

  await page.route(/\/api\/v1\/llm\/providers(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      providers: [{ name: "server-default", models: ["server-default-model"] }],
    })
  })

  await page.route(/\/api\/v1\/characters(?:\/search)?(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, [])
  })

  await page.route(/\/api\/v1\/chats(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { chats: [], total: 0 })
  })

  await page.route(/\/api\/v1\/chat\/conversations(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { conversations: [], total: 0 })
  })

  await page.route(/\/api\/v1\/media(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      items: [{ id: 101, title: "Grounded QA Notes", media_type: "document" }],
      total: 1,
    })
  })

  await page.route(/\/api\/v1\/notes(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      items: [{ id: "note-grounded-qa", title: "Grounded QA checklist" }],
      total: 1,
    })
  })
}

async function forceKnowledgeConnectionState(
  page: Page,
  knowledgeStatus: "ready" | "empty"
) {
  await page.waitForFunction(
    () => typeof (window as any).__tldw_useConnectionStore?.getState === "function"
  )
  await page.evaluate((status) => {
    const store = (window as any).__tldw_useConnectionStore
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
        knowledgeStatus: status,
        knowledgeLastCheckedAt: now,
        knowledgeError: null,
        configStep: "health",
        hasCompletedFirstRun: true,
      },
    })
  }, knowledgeStatus)
}

test.describe("Knowledge QA empty recovery", () => {
  test("WebUI shows add/index recovery when the backend reports no indexed sources", async ({ page }) => {
    await mockKnowledgeRecoveryApi(page, { hasWebSearch: false })
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })
    await expect(page.getByRole("heading", { name: /Ask Your Library/i })).toBeVisible()
    await forceKnowledgeConnectionState(page, "empty")

    await expect(page.getByText("No indexed library sources yet")).toBeVisible()
    await expect(
      page.getByRole("button", { name: "Add or index sources" }).last()
    ).toBeVisible()

    await page.getByLabel(/Search your knowledge base/i).fill("What does my library say?")
    await expect(page.getByRole("button", { name: /^Ask$/i })).toBeDisabled()
    await expect(
      page.getByText("Add or index library sources before asking Knowledge QA.")
    ).toBeVisible()
  })

  test("WebUI shows source-selection recovery when indexed sources exist but none are selected", async ({ page }) => {
    await mockKnowledgeRecoveryApi(page, { hasWebSearch: true })
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })
    await expect(page.getByRole("heading", { name: /Ask Your Library/i })).toBeVisible()
    await forceKnowledgeConnectionState(page, "ready")

    const webToggle = page.getByRole("button", {
      name: /Web fallback is currently/i,
    })
    await expect(webToggle).toBeVisible()
    if ((await webToggle.getAttribute("aria-pressed")) === "true") {
      await webToggle.click()
    }
    await page.getByRole("button", { name: /Open source scope and saved profiles/i }).click()
    const scopeDialog = page.getByRole("dialog", { name: "Source scope and profiles" })
    await expect(scopeDialog).toBeVisible()
    await scopeDialog.getByRole("button", { name: /Sources:/i }).click()
    for (const label of [
      "Documents & Media",
      "Notes",
      "Characters",
      "Chats",
      "Task Boards",
    ]) {
      const sourceOption = scopeDialog.getByRole("menuitemcheckbox", {
        name: new RegExp(label),
      })
      if ((await sourceOption.count()) === 0) continue
      if ((await sourceOption.first().getAttribute("aria-checked")) === "true") {
        await sourceOption.first().click()
      }
    }
    await scopeDialog.getByRole("button", { name: "Close source scope" }).click()

    await expect(page.getByText("No source categories selected")).toBeVisible()
    await expect(
      page.getByRole("button", { name: "Select source categories" }).last()
    ).toBeVisible()

    await page.getByLabel(/Search your knowledge base/i).fill("What does my library say?")
    await expect(page.getByRole("button", { name: /^Ask$/i })).toBeDisabled()
    await expect(
      page.getByText("Select source categories or enable web fallback before asking Knowledge QA.")
    ).toBeVisible()
  })
})
