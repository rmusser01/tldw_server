import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT Finding #1 (corrected): the chat model picker has no loading state, so during
 * the slow /api/v1/llm/models/metadata fetch it shows the terminal-sounding error
 * "No models available. Connect your server in Settings." against a reachable server.
 *
 * Correct behavior: while models are loading, show a loading affordance and NOT the
 * connect-server error; once loaded, real models appear.
 */
const WEB = "http://localhost:8080"
const SERVER = "http://127.0.0.1:8000"
const KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"

test("model picker shows a loading state (not a connect-server error) while models load", async ({ page }) => {
  test.setTimeout(120_000)

  let metaFinished = false
  page.on("requestfinished", (r) => {
    if (r.url().includes("/llm/models/metadata")) metaFinished = true
  })

  await seedAuth(page, { serverUrl: SERVER, apiKey: KEY })
  await page.goto(`${WEB}/chat`, { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })

  let sawConnectErrorWhileLoading = false
  let sawLoadingAffordance = false

  // Sample the dropdown while the metadata fetch is still in flight.
  for (let t = 0; t < 40 && !metaFinished; t++) {
    await page.getByTestId("model-selector").first().click().catch(() => {})
    await page.waitForTimeout(250)
    const body = await page.locator("body").innerText().catch(() => "")
    if (/Connect your server in Settings/i.test(body)) sawConnectErrorWhileLoading = true
    if ((await page.getByTestId("model-loading").count()) > 0) sawLoadingAffordance = true
    await page.keyboard.press("Escape").catch(() => {})
  }

  // After the fetch resolves, real models must be available.
  await expect
    .poll(async () => {
      await page.getByTestId("model-selector").first().click().catch(() => {})
      const body = await page.locator("body").innerText().catch(() => "")
      await page.keyboard.press("Escape").catch(() => {})
      return /gpt-4o|claude-|gemini|tldw:/i.test(body) && !/No models available/i.test(body)
    }, { timeout: 30_000, intervals: [1000] })
    .toBe(true)

  expect(sawConnectErrorWhileLoading, "must not show the connect-server error while models are loading").toBe(false)
  expect(sawLoadingAffordance, "must show a loading affordance while models load").toBe(true)
})
