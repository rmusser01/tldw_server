import type { Page } from "@playwright/test"
import { expect, seedAuth, test } from "./smoke.setup"

const bypassOnboarding = async (page: Page) => {
  await seedAuth(page, {
    authMode: "single-user",
    apiKey: "test-key-not-placeholder",
    allowOffline: false,
  })
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
      window.localStorage.setItem(
        "tldw-ui-mode",
        JSON.stringify({ state: { mode: "pro" }, version: 0 })
      )
    } catch {
      /* ignore */
    }
  })
}

const mockConversationContextAssets = async (page: Page) => {
  await page.route(/\/api\/v1\/config\/docs-info.*/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ docs_url: "http://localhost/docs" }),
    })
  })

  await page.route(/\/api\/v1\/health.*/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    })
  })

  await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        profile_version: "2026-05-09T00:00:00Z",
        preferences: {},
      }),
    })
  })

  await page.route(
    "**/api/v1/characters/world-books**",
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          world_books: [
            {
              id: 3,
              name: "Echo Vault Worldbook",
              enabled: true,
              entry_count: 1,
            },
          ],
        }),
      })
    }
  )

  await page.route(
    "**/api/v1/chat/dictionaries**",
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          dictionaries: [
            {
              id: 7,
              name: "Echo Vault Dictionary",
              is_active: true,
              entry_count: 1,
            },
          ],
        }),
      })
    }
  )
}

const dismissBlockingDialogs = async (page: Page) => {
  await page
    .locator(".ant-modal-wrap, nextjs-portal, [data-nextjs-dialog-overlay]")
    .evaluateAll((elements) => {
      for (const element of elements) {
        if (element instanceof HTMLElement) element.style.display = "none"
      }
    })
    .catch(() => {})
}

test.describe("conversation context popover", () => {
  test("blank sidepanel chat exposes character, worldbook, and dictionary slots", async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await mockConversationContextAssets(page)

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    await dismissBlockingDialogs(page)

    const trigger = page.getByTestId("conversation-context-trigger")
    await expect(trigger).toBeVisible({ timeout: 30_000 })
    await trigger.click()

    await expect(page.getByText("Conversation context")).toBeVisible()
    await expect(page.getByText("Character").first()).toBeVisible()
    await expect(page.getByText("Worldbooks").first()).toBeVisible()
    await expect(page.getByText("Dictionaries").first()).toBeVisible()
    await expect(page.getByLabel("Echo Vault Worldbook")).toBeVisible()
    await expect(page.getByLabel("Echo Vault Dictionary")).toBeVisible()
    await expect(page.getByLabel("Echo Vault Worldbook")).toBeDisabled()
    await expect(page.getByText(/character-card exclusive/i)).toHaveCount(0)
  })

  test("mobile sidepanel keeps the context trigger usable", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await mockConversationContextAssets(page)
    await page.setViewportSize({ width: 360, height: 800 })

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})
    await dismissBlockingDialogs(page)

    const trigger = page.getByTestId("conversation-context-trigger")
    await expect(trigger).toBeVisible({ timeout: 30_000 })
    await trigger.click()
    await expect(page.getByText("Conversation context")).toBeVisible()

    const triggerOverflow = await trigger.evaluate((el) => {
      const parent = el.parentElement
      if (!parent) return false
      return el.scrollWidth > parent.clientWidth
    })
    expect(triggerOverflow).toBe(false)
  })
})
