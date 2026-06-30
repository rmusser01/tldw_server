/**
 * Phase 6 Character Chat signoff.
 *
 * This suite intentionally uses the real WebUI route and backend fixture. It
 * does not mock Character Chat APIs; provider-unconfigured environments should
 * still expose the setup/recovery surfaces without layout overflow.
 */
import { type Page } from "@playwright/test"
import {
  assertNoCriticalErrors,
  expect,
  skipIfServerUnavailable,
  test,
} from "../../utils/fixtures"
import {
  expectNoHorizontalOverflow,
  waitForConnection,
  waitForVisualSettle,
} from "../../utils/helpers"

type ViewportTarget = {
  label: "desktop" | "tablet" | "mobile"
  width: number
  height: number
}

const VIEWPORTS: ViewportTarget[] = [
  { label: "desktop", width: 1440, height: 900 },
  { label: "tablet", width: 768, height: 1024 },
  { label: "mobile", width: 390, height: 844 },
]

async function openRolePlaySetup(page: Page): Promise<void> {
  const directSetupButtons = page.getByTestId("composer-role-play-setup")
  const directSetupCount = await directSetupButtons.count()
  let openedDirectly = false
  for (let index = 0; index < directSetupCount; index += 1) {
    const button = directSetupButtons.nth(index)
    if (await button.isVisible().catch(() => false)) {
      await button.click()
      openedDirectly = true
      break
    }
  }

  if (!openedDirectly) {
    await page.getByRole("button", { name: "More options" }).first().click()
    await page
      .getByRole("button", { name: "Role-play setup", exact: true })
      .click()
  }

  const dialog = page.getByRole("dialog", { name: "Role-play setup" })
  await expect(dialog).toBeVisible({
    timeout: 15_000,
  })
  await expect
    .poll(
      async () => {
        const box = await dialog.boundingBox()
        const viewportWidth = await page.evaluate(
          () => document.documentElement.clientWidth,
        )
        if (!box) return false
        return box.x >= -1 && box.x + box.width <= viewportWidth + 1
      },
      {
        timeout: 5_000,
        message: "Role-play setup drawer should settle inside the viewport",
      },
    )
    .toBe(true)
}

async function expectCharacterSessionsReachable(page: Page): Promise<void> {
  const sessions = page.getByRole("region", {
    name: "Character chat sessions",
  })
  if (await sessions.isVisible().catch(() => false)) {
    return
  }

  const showPanels = page.getByRole("button", {
    name: "Show cockpit panels",
  })
  if (await showPanels.isVisible().catch(() => false)) {
    await showPanels.click()
  }

  const contextTab = page.getByRole("tab", { name: "Context" })
  if (await contextTab.isVisible().catch(() => false)) {
    await contextTab.click()
  }

  await expect(sessions).toBeVisible({ timeout: 30_000 })
}

test.describe("Character Chat Phase 6 signoff", () => {
  for (const viewport of VIEWPORTS) {
    test(`character mode setup and recovery surfaces fit ${viewport.label}`, async ({
      authedPage: page,
      diagnostics,
      serverInfo,
    }) => {
      skipIfServerUnavailable(serverInfo)

      await page.setViewportSize({
        width: viewport.width,
        height: viewport.height,
      })
      await page.goto("/chat?mode=character", {
        waitUntil: "domcontentloaded",
      })
      await waitForConnection(page)

      await expect(
        page.getByTestId("playground-active-chat-mode"),
      ).toContainText("Character Chat", { timeout: 30_000 })
      await expectCharacterSessionsReachable(page)
      await expect(
        page.getByTestId("character-chat-readiness-panel"),
      ).toBeVisible({ timeout: 30_000 })
      await expect(page.getByPlaceholder(/type a message/i)).toBeVisible({
        timeout: 30_000,
      })
      await expectNoHorizontalOverflow(page, `${viewport.label} character chat`)

      await openRolePlaySetup(page)
      await waitForVisualSettle(page)
      await expectNoHorizontalOverflow(
        page,
        `${viewport.label} role-play setup drawer`,
      )

      await assertNoCriticalErrors(diagnostics)
    })
  }
})
