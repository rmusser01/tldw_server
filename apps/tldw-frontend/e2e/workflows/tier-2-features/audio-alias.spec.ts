/**
 * Audio alias E2E Tests (Tier 2)
 *
 * Locks /audio as a UI-free alias to the canonical /speech route.
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/audio-alias.spec.ts
 */
import { assertNoCriticalErrors, expect, test } from "../../utils/fixtures"
import { waitForAppShell } from "../../utils/helpers"

const LOAD_TIMEOUT = 30_000

test.describe("Audio Alias", () => {
  test("opens the canonical speech playground and preserves route context", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/audio?source=e2e-alias#voice", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT,
    })

    await authedPage.waitForURL(
      (url) =>
        url.pathname === "/speech" &&
        url.searchParams.get("source") === "e2e-alias" &&
        url.hash === "#voice",
      { timeout: LOAD_TIMEOUT }
    )
    await waitForAppShell(authedPage, LOAD_TIMEOUT)

    await expect(authedPage.getByRole("heading", { name: /^Speech Playground$/i })).toBeVisible({
      timeout: LOAD_TIMEOUT,
    })
    await expect(authedPage.getByRole("radio", { name: /^Round-trip$/i })).toBeVisible()
    await expect(authedPage.getByRole("radio", { name: /^Speak$/i })).toBeVisible()
    await expect(authedPage.getByRole("radio", { name: /^Listen$/i })).toBeVisible()
    await expect(authedPage.getByTestId("route-redirect-panel")).toHaveCount(0)

    await assertNoCriticalErrors(diagnostics)
  })
})
