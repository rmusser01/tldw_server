import { test, expect, seedAuth } from "./smoke.setup"
import { waitForAppShell } from "../utils/helpers"

const LOAD_TIMEOUT = 30_000

test.describe("Stage 4 accessibility controls", () => {
  test("header search trigger opens command palette and restores focus on Escape", async ({
    page
  }) => {
    await seedAuth(page)

    await page.goto("/chat", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const trigger = page.getByRole("button", { name: "Open command palette" })
    await expect(trigger).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(trigger).toHaveAttribute("title", "Search")

    await trigger.click()

    const palette = page.getByRole("dialog", { name: /command palette/i })
    await expect(palette).toBeVisible({ timeout: LOAD_TIMEOUT })

    const searchInput = page.getByRole("textbox", { name: "Search commands" })
    await expect(searchInput).toBeFocused()

    await page.keyboard.press("Escape")
    await expect(palette).toHaveCount(0)
    await expect(trigger).toBeFocused()
  })

  test("document workspace pane toggles expose explicit accessible labels", async ({
    page
  }) => {
    await seedAuth(page)

    await page.goto("/document-workspace", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const leftToggle = page.getByTestId("document-workspace-toggle-left")
    const rightToggle = page.getByTestId("document-workspace-toggle-right")

    await expect(leftToggle).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(leftToggle).toHaveAccessibleName(/(expand|collapse) sidebar/i)
    await expect(rightToggle).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(rightToggle).toHaveAccessibleName(/(expand|collapse) chat panel/i)

    const leftLabelBefore = await leftToggle.getAttribute("aria-label")
    await leftToggle.click()
    const leftLabelAfter = await leftToggle.getAttribute("aria-label")
    expect(leftLabelAfter).not.toBe(leftLabelBefore)

    const rightLabelBefore = await rightToggle.getAttribute("aria-label")
    await rightToggle.click()
    const rightLabelAfter = await rightToggle.getAttribute("aria-label")
    expect(rightLabelAfter).not.toBe(rightLabelBefore)
  })

  test("settings beta badge dismissal persists across reloads", async ({
    page
  }) => {
    await seedAuth(page)
    await page.addInitScript(() => {
      try {
        const resetKey = "__tldw_stage4_beta_badge_reset_done"
        if (sessionStorage.getItem(resetKey) === "1") {
          return
        }
        localStorage.removeItem("tldw:settings:hide-beta-badges")
        sessionStorage.setItem(resetKey, "1")
      } catch {}
    })

    await page.goto("/settings", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    const badges = page.locator('[data-testid="settings-navigation"] .ant-tag')
    const toggle = page.getByTestId("settings-beta-badges-toggle")

    await expect(toggle).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(toggle).toHaveText(/hide beta badges/i)
    await expect(badges.first()).toBeVisible({ timeout: LOAD_TIMEOUT })

    await toggle.click()
    await expect(toggle).toHaveText(/show beta badges/i)
    await expect(badges).toHaveCount(0)

    await page.reload({ waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })
    await waitForAppShell(page, LOAD_TIMEOUT)
    await expect(toggle).toHaveText(/show beta badges/i)
    await expect(badges).toHaveCount(0)

    await toggle.click()
    await expect(toggle).toHaveText(/hide beta badges/i)
    await expect(badges.first()).toBeVisible({ timeout: LOAD_TIMEOUT })
  })
})
