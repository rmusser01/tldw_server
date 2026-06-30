import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * /media-multi (MediaReviewPage) — core multi-select flow regression coverage.
 * Selecting result rows must surface the batch toolbar with its actions, and the
 * selected-items drawer must close on Escape (guards the app-wide Escape fix on
 * this page). Auth + base URL come from the env-driven smoke config.
 */
test("selecting items shows the batch toolbar with its actions", async ({ page }) => {
  test.setTimeout(90_000)
  await seedAuth(page)
  await page.goto("/media-multi", { waitUntil: "domcontentloaded" })
  await page.getByTestId("media-review-results-list").first().waitFor({ state: "visible", timeout: 30_000 })

  const rows = page.getByTestId("media-review-result-row")
  await expect(rows.first()).toBeVisible()

  // Each row toggles selection on Space (keyboard a11y path).
  for (let i = 0; i < 2; i++) {
    await rows.nth(i).focus()
    await page.keyboard.press("Space")
  }
  await expect(rows.nth(0)).toHaveAttribute("aria-selected", "true")
  await expect(rows.nth(1)).toHaveAttribute("aria-selected", "true")

  const bar = page.getByTestId("media-multi-batch-toolbar")
  await expect(bar).toBeVisible()
  await expect(bar).toContainText(/2 selected/i)
  await expect(page.getByTestId("media-multi-batch-add-tags")).toBeVisible()
  await expect(page.getByTestId("media-multi-batch-export")).toBeVisible()
  await expect(page.getByTestId("media-multi-batch-trash")).toBeVisible()
})

test("Escape closes the selected-items drawer", async ({ page }) => {
  test.setTimeout(90_000)
  await seedAuth(page)
  await page.goto("/media-multi", { waitUntil: "domcontentloaded" })
  await page.getByTestId("media-review-results-list").first().waitFor({ state: "visible", timeout: 30_000 })

  const rows = page.getByTestId("media-review-result-row")
  await rows.first().focus()
  await page.keyboard.press("Space")

  const viewSelected = page.getByTestId("view-selected-items-button").first()
  await viewSelected.click()
  await expect(page.getByTestId("selected-items-drawer")).toBeVisible()

  await page.keyboard.press("Escape")
  await expect(page.getByTestId("selected-items-drawer")).toBeHidden()
})
