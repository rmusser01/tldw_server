import { expect, test, type Route } from "@playwright/test"

/**
 * XS-15 (#3101), coordinator's decision: in the side panel the textarea and
 * Send always stay fully in view; optional rows (hint card, Runtime chip,
 * queue, chips) scroll in a region above them. Pinned at 400x600 in Pro, on
 * the WebUI's side-panel debug route (the same route component the
 * extension uses).
 *
 * Offline: the tldw API is stubbed with page.route. Start the WebUI with
 * NEXT_PUBLIC_API_URL pointing at TLDW_SERVER_URL (default
 * http://127.0.0.1:8000); nothing needs to listen there. Set
 * TLDW_E2E_CHROMIUM to a Chromium binary if Playwright's own isn't installed.
 */
const API = (process.env.TLDW_SERVER_URL || "http://127.0.0.1:8000").replace(/\/+$/, "")

test.use({
  viewport: { width: 400, height: 600 },
  ...(process.env.TLDW_E2E_CHROMIUM
    ? { launchOptions: { executablePath: process.env.TLDW_E2E_CHROMIUM }, video: "off" as const, trace: "off" as const }
    : {})
})

const stubTldwApi = async (route: Route, origin: string) => {
  const request = route.request()
  const headers = {
    "access-control-allow-origin": origin,
    "access-control-allow-credentials": "true",
    "access-control-allow-headers": "content-type, x-api-key, authorization, idempotency-key, x-request-id",
    "access-control-allow-methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS"
  }
  if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers })
  const path = new URL(request.url()).pathname
  const json = (body: unknown, status = 200) =>
    route.fulfill({ status, headers: { ...headers, "content-type": "application/json" }, body: JSON.stringify(body) })
  if (path.startsWith("/api/v1/health")) return json({ status: "ok", ok: true })
  if (path === "/api/v1/llm/models/metadata")
    return json([{ id: "mock-model", name: "Mock Model", provider: "mock", context_length: 4096, capabilities: ["chat"] }])
  if (path === "/api/v1/llm/models") return json(["mock-model"])
  if (path === "/openapi.json")
    return json({
      openapi: "3.0.0",
      info: { version: "stub" },
      paths: { "/api/v1/health": {}, "/api/v1/chat/completions": {}, "/api/v1/llm/models": {}, "/api/v1/llm/models/metadata": {} }
    })
  return request.method() === "GET" ? json({}) : json({ detail: "not found" }, 404)
}

test("keeps the textarea and Send fully in view in Pro at 400x600", async ({ page, baseURL }) => {
  const origin = new URL(baseURL || "http://localhost:8080").origin
  await page.route(`${API}/**`, (route) => stubTldwApi(route, origin))
  await page.addInitScript((serverUrl) => {
    localStorage.setItem("tldwConfig", JSON.stringify({
      serverUrl, authMode: "single-user", authSource: "manual", credentialSource: "manual",
      apiKeyPersistence: "device", apiKeyServerOrigin: serverUrl, apiKey: "test-key", accessToken: ""
    }))
    for (const [key, value] of Object.entries({
      isMigrated: "true", serverUrl, tldwServerUrl: serverUrl, authMode: "single-user",
      __tldw_first_run_complete: "true", assistant_setup_dismissed: "true", __tldw_allow_offline: "true"
    })) localStorage.setItem(key, value)
    localStorage.setItem("tldw-ui-mode", JSON.stringify({ state: { mode: "pro" }, version: 1 }))
  }, API)

  await page.goto("/__debug__/sidepanel-chat")
  const dock = page.getByTestId("sidepanel-composer-dock")
  const input = dock.getByTestId("chat-input")
  const send = dock.getByTestId("chat-send")
  await expect(input).toBeVisible({ timeout: 120_000 })
  await expect(send).toBeVisible()
  await expect(page.getByTestId("sidepanel-runtime-chip")).toBeAttached()
  // The debug route sits the panel 40px down the page; bring its bottom into view, as in the side panel.
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))

  for (const control of [input, send]) {
    await expect(control).toBeInViewport({ ratio: 1 })
    const box = await control.boundingBox()
    const dockBox = await dock.boundingBox()
    expect(box && dockBox).toBeTruthy()
    expect(box!.y).toBeGreaterThanOrEqual(dockBox!.y - 1)
    expect(box!.y + box!.height).toBeLessThanOrEqual(dockBox!.y + dockBox!.height + 1)
    // Nothing (the optional region, the conversation) covers it.
    const covered = await control.evaluate((element) => {
      const rect = element.getBoundingClientRect()
      const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2)
      return !(hit && (hit === element || element.contains(hit)))
    })
    expect(covered).toBe(false)
  }
})
