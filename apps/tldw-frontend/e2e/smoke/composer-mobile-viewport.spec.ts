import type { Page } from "@playwright/test"
import { expect, seedAuth, test } from "./smoke.setup"

/**
 * Mobile-viewport smoke for the current composer direction. The sidepanel
 * uses V5 as the mobile reference; V1/V3 remain selectable layouts, but they
 * are not the standard we design or regress against for narrow screens.
 *
 * We don't snapshot pixels. Instead we assert structural invariants:
 *   - the V5 compact mobile rows render at 360px
 *   - the textarea keeps usable width
 *   - desktop command affordances do not leak into mobile
 *   - the composer wrapper itself does not exceed its container
 */

const bypassOnboarding = async (page: Page) => {
  await seedAuth(page)
  await page.addInitScript(() => {
    window.localStorage.setItem("playgroundComposerOptionsExpanded", "false")
  })
}

const mockComposerProfile = async (page: Page) => {
  await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
    const method = route.request().method()
    if (method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          profile_version: "2026-04-20T00:00:00Z",
          preferences: {},
        }),
      })
      return
    }
    if (method === "PATCH") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ applied: [], skipped: [] }),
      })
      return
    }
    await route.continue()
  })
}

const setV5 = async (page: Page) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("tldw:composerVariant", "v5")
  })
}

const assertWrapperDoesNotOverflow = async (page: Page) => {
  const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
  const overflow = await wrapper.evaluate((el) => {
    const parent = el.parentElement
    if (!parent) return false
    return el.scrollWidth > parent.clientWidth
  })
  expect(overflow).toBe(false)
}

test.describe("composer · mobile viewport", () => {
  test("sidepanel: V5 at 360px keeps a usable compact composer", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await mockComposerProfile(page)
    await setV5(page)
    await page.setViewportSize({ width: 360, height: 800 })

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page
      .waitForLoadState("networkidle", { timeout: 30_000 })
      .catch(() => {})

    await expect(
      page.locator('[data-testid="chat-header-sidebar-toggle"]')
    ).toHaveCount(0)
    await expect(
      page.locator('[data-testid="chat-header-companion-home"]')
    ).toHaveCount(0)
    await expect(page.locator('[data-testid="chat-header"]')).toHaveCount(1)

    const emptyState = page.locator('[data-testid="chat-empty-connected"]')
    await expect(emptyState).toBeVisible()
    await expect(emptyState.getByText("Connected", { exact: true })).toHaveCount(
      0
    )
    await expect(page.locator('[data-testid="chat-suggestion-3"]')).toHaveCount(0)

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v5']")).toBeVisible()
    await expect(wrapper.locator('[data-testid="v5-mobile-composer"]')).toBeVisible()
    await expect(wrapper.locator('[data-testid="v5-mobile-text-row"]')).toBeVisible()
    await expect(wrapper.locator('[data-testid="v5-mobile-action-row"]')).toBeVisible()
    await expect(wrapper.getByText("⌘K")).toHaveCount(0)
    await expect(
      wrapper.locator('[data-testid="chat-upload-image-inline"]')
    ).toHaveCount(0)
    await expect(
      wrapper.locator('[data-testid="chat-attach-document-inline"]')
    ).toBeVisible()

    const metaText = await wrapper
      .locator('[data-testid="v5-mobile-meta-row"]')
      .innerText()
    expect(metaText).not.toContain("MDL")
    expect(metaText).not.toContain("—")

    const chatInput = wrapper.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    const inputBox = await chatInput.boundingBox()
    expect(inputBox?.width ?? 0).toBeGreaterThan(220)

    await assertWrapperDoesNotOverflow(page)
  })

  test("/chat: V5 tablet viewport fits without composer overflow", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await mockComposerProfile(page)
    await setV5(page)
    await page.setViewportSize({ width: 768, height: 800 })

    await page.goto("/chat?nextgenComposer=1")
    await page
      .waitForLoadState("networkidle", { timeout: 30_000 })
      .catch(() => {})

    await page.evaluate(() => {
      document
        .querySelectorAll("nextjs-portal, [role='dialog']")
        .forEach((el) => {
          if (el instanceof HTMLElement) el.style.display = "none"
        })
    })

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator("[data-variant='v5']")).toBeVisible()
    await expect(wrapper.locator('[data-testid="chat-input"]')).toBeVisible()

    await assertWrapperDoesNotOverflow(page)
  })
})
