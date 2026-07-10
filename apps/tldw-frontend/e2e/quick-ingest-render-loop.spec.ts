import { test, expect, seedAuth } from "./smoke/smoke.setup"

test("Quick Ingest defaults flow does not trigger a Maximum update depth render loop", async ({
  page,
}) => {
  test.setTimeout(90_000)
  const renderLoopErrors: string[] = []

  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      /Maximum update depth exceeded/i.test(message.text())
    ) {
      renderLoopErrors.push(message.text().slice(0, 240))
    }
  })
  page.on("pageerror", (error) => {
    if (/Maximum update depth exceeded/i.test(error.message)) {
      renderLoopErrors.push(error.message)
    }
  })

  await page.route("**/api/v1/llm/models/metadata**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [], total: 0 }),
    })
  })

  await seedAuth(page)
  await page.goto("/", { waitUntil: "domcontentloaded" })

  const quickIngestTrigger = page.getByRole("button", { name: /^Quick Ingest$/i }).first()
  await expect(quickIngestTrigger).toBeVisible({ timeout: 30_000 })
  await quickIngestTrigger.click()

  const dialog = page.getByRole("dialog", { name: /Quick Ingest/i }).first()
  await expect(dialog).toBeVisible({ timeout: 30_000 })
  const urlInput = dialog.locator("textarea").first()
  await urlInput.fill("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
  await dialog.getByRole("button", { name: /Add URLs/i }).click()
  await expect(dialog).toContainText("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

  await dialog.getByRole("button", { name: /Use defaults/i }).click()
  await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {})

  await expect(page.locator("body")).not.toContainText("Runtime Error")
  expect(renderLoopErrors, "no Maximum update depth render loop").toEqual([])
})
