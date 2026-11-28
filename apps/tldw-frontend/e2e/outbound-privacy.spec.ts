import { expect, test } from "@playwright/test"

import { fixtureSetting, loopbackOrigin, observeOutbound } from "./utils/outbound-observer"

const apiOrigin = loopbackOrigin(18081)
const webOrigin = loopbackOrigin(18082)

test("shipped setup saves a local setting", async ({ context, page }) => {
  const recorder = await observeOutbound(context, [apiOrigin])
  try {
    const guard = page.waitForResponse(`${apiOrigin}/static/setup/js/setup-guard.js`)
    const script = page.waitForResponse(`${apiOrigin}/static/setup/js/setup.js`)
    const document = await page.goto(`${apiOrigin}/setup`)
    expect(document?.status()).toBe(200)
    expect((await guard).ok()).toBe(true)
    expect((await script).ok()).toBe(true)
    await expect(page).toHaveTitle("TLDW Server - First-Time Setup")

    await expect(page.locator("#guidedWizard")).toBeVisible()
    await page.locator("#wizardSkip").click()
    const field = page.locator('[data-section="Files"][data-key="inline_max_bytes"]')
    await expect(field).toHaveValue("262144")
    await field.fill("262145")
    const saved = page.waitForResponse(response =>
      response.url() === `${apiOrigin}/api/v1/setup/config` &&
      response.request().method() === "POST"
    )
    await page.locator("#saveChanges").click()
    expect((await saved).ok()).toBe(true)
    await expect(page.locator("#actionMessage")).toContainText("Configuration saved")
  } finally {
    await recorder.finish()
  }
})

test("production WebUI renders search results without external assets", async ({
  browser,
  context,
  page
}) => {
  const recorder = await observeOutbound(context, [webOrigin])
  try {
    // This separate context proves the passive observer catches an actual image request.
    const probeContext = await browser.newContext()
    try {
      const probe = await observeOutbound(probeContext, [webOrigin])
      const probePage = await probeContext.newPage()
      await probePage.setContent('<img src="http://127.0.0.1:18085/outbound-tripwire">')
      await expect.poll(probe.unexpectedRequests).toBe(1)
      recorder.noteTripwire()
    } finally {
      await probeContext.close()
    }

    const sessionCookie = fixtureSetting("OUTBOUND_SESSION_COOKIE")
    await page.goto(`${webOrigin}/chat`, { waitUntil: "domcontentloaded" })
    await expect.poll(async () =>
      (await context.cookies(`${webOrigin}/api/v1/auth/sessions`))
        .some(cookie => cookie.name === sessionCookie && cookie.path === "/api" && cookie.httpOnly),
      { timeout: 30_000 }
    ).toBe(true)
    const sessionStatus = await page.evaluate(async () =>
      (await fetch("/api/v1/auth/sessions", { credentials: "same-origin", cache: "no-store" })).status
    )
    expect(sessionStatus).toBe(200)
    const browserState = await page.evaluate(() => JSON.stringify({
      local: { ...localStorage },
      session: { ...sessionStorage },
      cookies: document.cookie
    }))
    expect(browserState.includes(fixtureSetting("OUTBOUND_SYNTHETIC_API_KEY"))).toBe(false)

    const snippet = "A local fixture snippet for the outbound privacy check."
    await page.route(`${webOrigin}/api/v1/research/websearch`, async route => {
      if (route.request().method() !== "POST") {
        await route.fallback()
        return
      }
      await route.fulfill({
        json: {
          results: [{ title: "Outbound fixture result", url: "https://result.invalid/article", snippet }]
        }
      })
    })
    await page.goto(`${webOrigin}/research-workspace`, { waitUntil: "domcontentloaded" })
    await page.locator("#workspace-sources-panel")
      .getByRole("button", { name: /^add( sources)?$/i }).click()
    const addSources = page.getByRole("dialog", { name: "Add Sources" })
    await addSources.getByRole("tab", { name: "Search Server" }).click()
    await addSources.getByRole("textbox", { name: "Search the web" }).fill("fixture")
    await addSources.getByRole("button", { name: "Search", exact: true }).click()
    await expect(addSources.getByText(snippet)).toBeVisible()
    await expect(addSources.getByTestId("search-result-icon")).toHaveAttribute("aria-hidden", "true")
  } finally {
    await recorder.finish()
  }
})
