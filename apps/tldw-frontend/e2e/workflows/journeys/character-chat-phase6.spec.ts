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
    // Mobile cockpit deliberately hides the full toolbar. Its visible Buddy
    // entry keeps conversation Persona and behavior settings reachable.
    await page.getByRole("button", { name: "Buddy & Persona", exact: true }).click()
    await page
      .getByRole("dialog", { name: "Buddy & Persona Management" })
      .getByRole("button", { name: "Edit conversation Persona & behavior", exact: true })
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

  // Rail visibility is independent of focus mode and defaults to collapsed.
  const restoreContext = page.getByRole("button", {
    name: "Restore context sidechannel", exact: true,
  })
  if (await restoreContext.isVisible()) {
    await restoreContext.click()
  } else {
    await page.getByRole("tab", {
      name: /^(Context|Restore context sidechannel)$/,
    }).click()
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

      // Mobile starts in focus mode. The same toggle enters focus on desktop,
      // so only activate it when its current pressed state means exit focus.
      const layoutToggle = page.getByTestId("playground-chat-layout-mode-trigger")
      await expect(layoutToggle).toBeVisible()
      if ((await layoutToggle.getAttribute("aria-pressed")) === "true") {
        await layoutToggle.click()
      }
      await expect(layoutToggle).toHaveAttribute("aria-pressed", "false")

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
