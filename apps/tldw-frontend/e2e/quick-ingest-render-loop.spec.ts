import type { Locator, Page, TestInfo } from "@playwright/test"
import { test, expect, seedAuth } from "./smoke/smoke.setup"
import { installBrowserFixture } from "./utils/media-ux-fixture"

test.describe.configure({ timeout: 120_000 })

async function openQuickIngest(page: Page): Promise<Locator> {
  const cors = {
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
  }
  await page.route("**/openapi.json", route => route.fulfill({
    json: {
      openapi: "3.1.0",
      info: { title: "Quick Ingest fixture", version: "1" },
      paths: { "/api/v1/media/": { get: {} }, "/api/v1/media/search": { get: {} } },
    },
    headers: cors,
  }))
  await page.route("**/api/v1/**", async route => {
    const path = new URL(route.request().url()).pathname
    let json: unknown = { status: "ok" }
    if (path.endsWith("/media")) {
      json = { items: [], pagination: { page: 1, results_per_page: 20, total_items: 0, total_pages: 0 } }
    } else if (path.includes("/llm/models")) {
      json = { models: [], total: 0 }
    } else if (path.includes("/providers")) {
      json = { providers: [], any_configured: false }
    }
    await route.fulfill({ json, headers: cors })
  })
  await seedAuth(page)
  await page.goto("/media", { waitUntil: "domcontentloaded" })
  const trigger = page.getByRole("button", { name: "Add media", exact: true })
  await trigger.click({ timeout: 60_000 })
  const dialog = page.getByRole("dialog", { name: /quick ingest/i })
  await expect(dialog).toBeVisible()
  return dialog
}

async function addLinks(dialog: Locator, links: string[]): Promise<void> {
  await dialog.getByRole("button", { name: "Links", exact: true }).click()
  await dialog.getByRole("textbox", { name: /URLs input/i }).fill(links.join("\n"))
  await dialog.getByRole("button", { name: /add urls/i }).click()
}

async function capture(page: Page, testInfo: TestInfo, name: string): Promise<void> {
  const path = testInfo.outputPath(`${name}.png`)
  await page.screenshot({ path, animations: "disabled" })
  await testInfo.attach(name, { path, contentType: "image/png" })
}

async function expectPhoneLayout(dialog: Locator, save: Locator, viewport: { width: number; height: number }): Promise<void> {
  const surface = await dialog.locator(".qi-web-surface").boundingBox()
  expect(surface).not.toBeNull()
  expect(Math.abs(surface!.x)).toBeLessThanOrEqual(1)
  expect(Math.abs(viewport.width - surface!.x - surface!.width)).toBeLessThanOrEqual(1)
  expect(Math.abs(surface!.y)).toBeLessThanOrEqual(1)
  expect(Math.abs(viewport.height - surface!.y - surface!.height)).toBeLessThanOrEqual(1)

  const buttonWidth = await save.evaluate(button => button.getBoundingClientRect().width)
  const footerWidth = await dialog.locator(".qi-web-footer").evaluate(footer => {
    const style = getComputedStyle(footer)
    return footer.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
  })
  expect(Math.abs(buttonWidth - footerWidth)).toBeLessThanOrEqual(1)

  const sourceRows = dialog.getByRole("region", { name: "Queued sources" }).getByRole("listitem")
  await expect(sourceRows).toHaveCount(3)
  for (const row of await sourceRows.all()) {
    const rowEdge = await row.evaluate(element => element.getBoundingClientRect().right)
    const removeEdge = await row.getByRole("button", { name: /^Remove / })
      .evaluate(button => button.getBoundingClientRect().right)
    expect(Math.abs(rowEdge - removeEdge)).toBeLessThanOrEqual(1)
  }
}

for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
  test(`Quick Ingest repairs a mixed queue and keeps the commit action visible at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    const renderLoopErrors: string[] = []
    page.on("console", message => {
      if (message.type() === "error" && /maximum update depth exceeded/i.test(message.text())) {
        renderLoopErrors.push(message.text())
      }
    })
    page.on("pageerror", error => {
      if (/maximum update depth exceeded/i.test(error.message)) renderLoopErrors.push(error.message)
    })

    const dialog = await openQuickIngest(page)
    const source = "https://example.org/field-notes"
    await addLinks(dialog, [source, source, "bad-link"])
    const counts = dialog.getByRole("status", { name: "Source counts" })
    await expect(counts).toHaveText("1 ready · 1 duplicate · 1 invalid")
    const save = dialog.getByRole("button", { name: "Save 1 source", exact: true })
    await expect(save).toBeDisabled()
    await expect(save).toBeInViewport({ ratio: 1 })
    const provider = dialog.getByRole("combobox", { name: "Analysis provider" })
    await expect(provider).toHaveAttribute("aria-describedby", /quick-ingest-analysis-provider-warning/)
    const actions = dialog.getByRole("group", { name: "Import actions" })
    await expect(dialog.getByText("AI analysis needs a provider", { exact: true })).toHaveCount(1)
    const choose = actions.getByRole("button", { name: "Choose provider", exact: true })
    const skip = actions.getByRole("button", { name: "Skip analysis for this import" })
    await expect(choose).toBeInViewport({ ratio: 1 })
    await expect(skip).toBeInViewport({ ratio: 1 })
    await capture(page, testInfo, "provider-required")
    const placeholder = dialog.getByText("Choose or enter a provider", { exact: true })
    const mutedColor = await dialog.locator("label[for='quick-ingest-analysis-provider']").evaluate(label => {
      const muted = getComputedStyle(label).getPropertyValue("--color-text-muted").trim()
      return "rgb(" + muted.split(/\s+/).join(", ") + ")"
    })
    await expect(placeholder).toHaveCSS("color", mutedColor)
    await expect(placeholder).toHaveCSS("opacity", "1")
    await choose.click()
    await expect(provider).toBeFocused()
    // Ant Design clips its internal input to the painted combobox border.
    await expect(dialog.locator(".ant-select").filter({ has: page.getByRole("combobox", { name: "Analysis provider" }) })).toBeInViewport({ ratio: 1 })

    await skip.click()
    await expect(save).toBeEnabled()
    await expect(save).toBeFocused()
    await counts.scrollIntoViewIfNeeded()
    if (viewport.width === 390) await expectPhoneLayout(dialog, save, viewport)
    await capture(page, testInfo, "mixed-queue")

    await dialog.getByRole("button", { name: "Change settings" }).click()
    const advanced = dialog.getByRole("button", { name: "Advanced options" })
    await advanced.click()
    await expect(dialog.getByRole("combobox", { name: "Auto chunking goal" })).toBeVisible()
    await expect(dialog.locator(".qi-web-footer")).toBeInViewport({ ratio: 1 })
    await expect(save).toBeInViewport({ ratio: 1 })
    await capture(page, testInfo, "expanded-settings")

    const preset = dialog.getByRole("combobox", { name: "Processing preset" })
    const destination = dialog.getByRole("combobox", { name: "Destination" })
    const overwrite = dialog.getByRole("switch", { name: /overwrite existing/i })
    await expect(overwrite).toBeVisible()
    await preset.selectOption("deep")
    await expect(destination).toHaveValue("draft")
    await expect(dialog.getByText("This preset changed the destination to Create review drafts.")).toBeVisible()
    await expect(overwrite).toHaveCount(0)
    await destination.selectOption("extract")
    await expect(overwrite).toHaveCount(0)
    await preset.selectOption("quick")
    await expect(destination).toHaveValue("media")
    await expect(overwrite).toBeVisible()

    await dialog.getByRole("button", { name: "Hide settings" }).click()
    await dialog.getByRole("button", { name: "Edit link" }).click()
    const edit = dialog.getByRole("textbox", { name: "Correct link" })
    await expect(edit).toHaveValue("bad-link")
    await edit.fill("https://example.org/repaired")
    await dialog.getByRole("button", { name: "Apply link" }).click()
    await expect(counts).toHaveText("2 ready · 1 duplicate")
    await expect(dialog.getByRole("button", { name: "Save 2 sources" })).toBeEnabled()
    await expect(page.locator("body")).not.toContainText("Runtime Error")
    expect(renderLoopErrors).toEqual([])
  })
}

test("Retry connection keeps an offline queue and its unsubmitted link", async ({ page }, testInfo) => {
  const webUrl = new URL(process.env.TLDW_WEB_URL || "http://localhost:8080").origin
  const fixture = await installBrowserFixture(page, { webUrl })
  let online = false
  await page.route("**/api/v1/health/live", route => {
    if (online || route.request().method() !== "GET") return route.fallback()
    return route.fulfill({
      status: 503, headers: { "access-control-allow-origin": "*" },
      json: { detail: "Server temporarily unavailable" },
    })
  })
  await page.goto(webUrl + "/media", { waitUntil: "domcontentloaded" })
  await expect(page.getByRole("button", { name: "Retry connection", exact: true }).first()).toBeVisible()
  await page.getByRole("button", { name: "Quick Ingest", exact: true }).click()
  const dialog = page.getByRole("dialog", { name: "Quick Ingest", exact: true })
  await addLinks(dialog, ["https://example.org/offline-notes"])
  const input = dialog.getByRole("textbox", { name: /URLs input/i })
  const unsent = "https://example.org/next-survey"
  await input.fill(unsent)

  const actions = dialog.getByRole("group", { name: "Import actions" })
  const save = actions.getByRole("button", { name: "Save 1 source" })
  const retry = actions.getByRole("button", { name: "Retry connection" })
  await expect(save).toBeDisabled()
  await expect(save).toHaveAccessibleDescription("Server offline")
  await expect(retry).toBeInViewport({ ratio: 1 })
  await expect(actions.getByRole("button", { name: "Choose provider" })).toHaveCount(0)
  await capture(page, testInfo, "offline-retry")
  const recovered = page.waitForResponse(response => response.url().endsWith("/api/v1/health/live") && response.ok())
  online = true
  await retry.click()
  await recovered
  await actions.getByRole("button", { name: "Skip analysis for this import" }).click()
  await expect(save).toBeEnabled()
  await expect(save).toBeFocused()
  await expect(input).toHaveValue(unsent)
  await expect(dialog.getByRole("status", { name: "Source counts" })).toHaveText("1 ready")
  expect(fixture.jobs.size).toBe(0)
})

test("Quick Ingest hydrates the saved Standard analysis provider from WebUI storage", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("quickIngestPresetConfigs", JSON.stringify({
      standard: { advancedValues: { api_name: "openai" } },
    }))
  })
  const dialog = await openQuickIngest(page)
  await addLinks(dialog, ["https://example.org/saved-provider"])
  await expect(dialog.getByText("AI analysis with openai", { exact: true })).toBeVisible()
  await expect(dialog.getByRole("button", { name: "Save 1 source" })).toBeEnabled()
  await dialog.getByRole("button", { name: "Change settings" }).click()
  await expect(dialog.getByRole("combobox", { name: "Analysis provider" })).toHaveValue("openai")
})
