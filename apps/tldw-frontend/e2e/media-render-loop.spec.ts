import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT (/media): ViewMediaPage logs "Maximum update depth exceeded" on load — an
 * infinite/excessive render loop. The reading-progress effect in useMediaSelection
 * re-runs on every `displayResults` reference change and calls setReadingProgressMap
 * with a fresh Map, so an unstable displayResults reference drives a setState loop
 * (and re-fires the per-item /media/{id}/progress fetches).
 *
 * Auth + base URL come from the env-driven smoke config.
 */
test("/media loads without a Maximum update depth render loop", async ({ page }) => {
  test.setTimeout(90_000)
  const renderLoopErrors: string[] = []
  page.on("console", (m) => {
    if (m.type() === "error" && /Maximum update depth exceeded/i.test(m.text())) {
      renderLoopErrors.push(m.text().slice(0, 120))
    }
  })

  await seedAuth(page)
  await page.goto("/media", { waitUntil: "domcontentloaded" })
  await page.getByTestId("media-search-input").first().waitFor({ state: "visible", timeout: 30_000 })

  // Wait for the page to actually settle (results render + network quiesces) rather
  // than a fixed sleep — the render loop, if present, manifests during this window.
  await page
    .getByTestId("media-results-list")
    .first()
    .waitFor({ state: "visible", timeout: 20_000 })
    .catch(() => {})
  await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {})

  expect(renderLoopErrors, "no Maximum update depth render loop on /media load").toEqual([])
})
