import type { Request, Response } from "@playwright/test"
import { test, expect } from "../utils/fixtures"
import { TEST_CONFIG } from "../utils/helpers"
import { UX_ROUTE_TIMEOUT_MS, waitForChatReady } from "./ux-routes"

const API_BUDGET_MS = 5_000
const NEXT_ROUTE_TIMEOUT_MS = UX_ROUTE_TIMEOUT_MS * 3

async function expectFastResponse(response: Response) {
  const label = new URL(response.url()).pathname
  expect(response.ok(), label).toBe(true)
  const duration = () => response.request().timing().responseEnd
  await expect.poll(duration, { message: label, timeout: API_BUDGET_MS }).toBeGreaterThanOrEqual(0)
  expect(duration(), label).toBeLessThan(API_BUDGET_MS)
}

// Runs first in the single-worker UX project, before any specs prime the backend.
test("#3135: cold Chat stays responsive and streams its first send", async ({
  authedPage: page,
  request,
}) => {
  test.setTimeout(240_000)
  const { serverUrl, webUrl, apiKey } = TEST_CONFIG

  // Let Next compile the HTML route before Chromium runs the client startup.
  const html = await request.get(`${webUrl}/chat`, { timeout: NEXT_ROUTE_TIMEOUT_MS })
  expect(html.ok()).toBe(true)

  const schemaUrl = `${serverUrl}/openapi.json`
  const completionUrl = `${serverUrl}/api/v1/chat/completions`
  const finishedBackendRequests: Request[] = []
  page.on("requestfinished", (finished) => {
    if (new URL(finished.url()).origin === serverUrl) finishedBackendRequests.push(finished)
  })

  // Start both probes with the browser's first schema request.
  const firstSchemaRequest = page.waitForRequest(
    (pending) => pending.method() === "GET" && pending.url() === schemaUrl,
    { timeout: NEXT_ROUTE_TIMEOUT_MS }
  )
  const startupProbes = firstSchemaRequest.then(async (schemaRequest) => {
    const [health, preflight, schema] = await Promise.all([
      request.get(`${serverUrl}/api/v1/health`, {
        headers: { "X-API-KEY": apiKey },
        timeout: API_BUDGET_MS,
      }),
      request.fetch(completionUrl, {
        method: "OPTIONS",
        headers: {
          Origin: new URL(webUrl).origin,
          "Access-Control-Request-Method": "POST",
          "Access-Control-Request-Headers": "content-type,x-api-key",
        },
        timeout: API_BUDGET_MS,
      }),
      schemaRequest.response(),
    ])

    expect(health.ok(), "Health response during the first schema request").toBe(true)
    expect(preflight.ok(), "Chat preflight during the first schema request").toBe(true)
    if (!schema) throw new Error("The first browser schema request received no response")
    await expectFastResponse(schema)
  })

  const catalogResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "GET" &&
      response.url() === `${serverUrl}/api/v1/llm/models/metadata`,
    { timeout: NEXT_ROUTE_TIMEOUT_MS }
  )
  const [navigation, catalog] = await Promise.all([
    page.goto("/chat", { waitUntil: "domcontentloaded", timeout: NEXT_ROUTE_TIMEOUT_MS }),
    catalogResponse,
    startupProbes,
  ])
  expect(navigation?.ok()).toBe(true)
  await expectFastResponse(catalog)
  await waitForChatReady(page)

  const prompt = "Say hello briefly."
  const send = page.getByRole("button", { name: /^send message$/i })
  await page.getByTestId("chat-input").fill(prompt)
  await expect(send).toBeEnabled()

  const reply = page.locator("article[aria-label*='Assistant message']")
  const [completion] = await Promise.all([
    page.waitForResponse(
      (response) => response.request().method() === "POST" && response.url() === completionUrl,
      { timeout: API_BUDGET_MS }
    ),
    expect(reply).toContainText("The mock provider returned a deterministic success response.", {
      timeout: API_BUDGET_MS,
    }),
    send.click({ timeout: API_BUDGET_MS }),
  ])
  expect(completion.ok()).toBe(true)
  expect(completion.headers()["content-type"]).toContain("text/event-stream")
  expect(completion.request().postDataJSON()).toMatchObject({
    stream: true,
    messages: expect.arrayContaining([expect.objectContaining({ role: "user", content: prompt })]),
  })
  for (const finished of finishedBackendRequests) {
    // The streaming completion is measured through the reply appearing above.
    if (finished === completion.request()) continue
    const label = `${finished.method()} ${new URL(finished.url()).pathname}`
    expect(finished.timing().responseEnd, label).toBeLessThan(API_BUDGET_MS)
  }
})
