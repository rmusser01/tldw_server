import { expect, test, type Page } from "@playwright/test"

/**
 * End-to-end proof that the `storage` event listener in
 * `useComposerVariantPreference` actually propagates a variant
 * change in tab B to tab A — without a reload.
 *
 * Two pages share the same Playwright browser context, which means
 * they share the same `localStorage` origin. A `storage` event
 * fires in tab A whenever any other tab in the same context writes
 * to a localStorage key.
 *
 * Note: due to a quirk of how same-context tabs synchronously share
 * storage in jsdom-style environments, real browsers fire `storage`
 * only on OTHER windows. We exercise the real-browser path.
 */

const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
    } catch {
      /* ignore */
    }
  })
}

const mockProfile = async (page: Page) => {
  await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
    const method = route.request().method()
    if (method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          profile_version: "2026-04-19T00:00:00Z",
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

test("composer · cross-tab sync: picker click in tab B updates /chat in tab A live", async ({
  browser,
}) => {
  test.setTimeout(120_000)

  // Single browser context = shared localStorage = `storage` events
  // fire across tabs the way they would for a real user.
  const context = await browser.newContext()
  const tabA = await context.newPage()
  const tabB = await context.newPage()

  await bypassOnboarding(tabA)
  await mockProfile(tabA)
  await bypassOnboarding(tabB)
  await mockProfile(tabB)

  // Tab A: open /chat with the flag on
  await tabA.goto("/chat?nextgenComposer=1")
  await tabA
    .waitForLoadState("networkidle", { timeout: 30_000 })
    .catch(() => {})

  const wrapperA = tabA.locator('[data-testid="nextgen-composer-wrapper"]')
  await expect(wrapperA).toBeVisible({ timeout: 30_000 })
  // Default variant is V1 on a fresh load.
  await expect(wrapperA.locator("[data-variant='v1']")).toBeVisible()

  // Tab B: open Settings, pick V3
  await tabB.goto("/settings/chat")
  const v3Card = tabB.getByRole("radio", { name: /split brief/i })
  await expect(v3Card).toBeVisible({ timeout: 15_000 })
  await v3Card.click()
  await expect(v3Card).toHaveAttribute("aria-checked", "true")

  // Tab A should now reflect V3 — without any reload — via the
  // storage event listener.
  await expect(wrapperA.locator("[data-variant='v3']")).toBeVisible({
    timeout: 5_000,
  })
  await expect(wrapperA.locator("[data-variant='v1']")).toHaveCount(0)

  await context.close()
})

test("composer · cross-tab sync: enable-toggle flip reveals composer live (no reload)", async ({
  browser,
}) => {
  test.setTimeout(120_000)

  const context = await browser.newContext()
  const tabA = await context.newPage()
  const tabB = await context.newPage()

  await bypassOnboarding(tabA)
  await mockProfile(tabA)
  await bypassOnboarding(tabB)
  await mockProfile(tabB)

  // Tab A: open /chat (no flag, toggle off) → legacy composer
  await tabA.goto("/chat")
  await tabA
    .waitForLoadState("networkidle", { timeout: 30_000 })
    .catch(() => {})
  await expect(
    tabA.locator('[data-testid="nextgen-composer-wrapper"]')
  ).toHaveCount(0)

  // Tab B: open Settings and flip the enable toggle on
  await tabB.goto("/settings/chat")
  const toggle = tabB.getByTestId("composer-enabled-toggle")
  await expect(toggle).toBeVisible({ timeout: 15_000 })
  await toggle.check()
  await expect(toggle).toBeChecked()

  // Tab A should auto-flip via the live storage listener in
  // useComposerEnabledPreference — no reload needed. Timeout is
  // generous because the ServerReadinessGate on /chat has its own
  // 15s fallthrough before the Playground form mounts.
  await expect(
    tabA.locator('[data-testid="nextgen-composer-wrapper"]')
  ).toBeVisible({ timeout: 30_000 })

  await context.close()
})

test("composer · cross-tab sync preserves draft text", async ({ browser }) => {
  test.setTimeout(120_000)

  const context = await browser.newContext()
  const tabA = await context.newPage()
  const tabB = await context.newPage()

  await bypassOnboarding(tabA)
  await mockProfile(tabA)
  await bypassOnboarding(tabB)
  await mockProfile(tabB)

  // Tab A: open /chat (V1 default), type a draft into the chat input.
  await tabA.goto("/chat?nextgenComposer=1")
  await tabA
    .waitForLoadState("networkidle", { timeout: 30_000 })
    .catch(() => {})

  const wrapperA = tabA.locator('[data-testid="nextgen-composer-wrapper"]')
  await expect(wrapperA).toBeVisible({ timeout: 30_000 })
  await expect(wrapperA.locator("[data-variant='v1']")).toBeVisible()

  // Hide the dev runtime overlay before typing
  await tabA.evaluate(() => {
    document.querySelectorAll("nextjs-portal, [role='dialog']").forEach(
      (el) => {
        if (el instanceof HTMLElement) el.style.display = "none"
      }
    )
  })

  const draft = "draft survives cross-tab switch"
  const chatInputA = wrapperA.locator('[data-testid="chat-input"]')
  await expect(chatInputA).toBeVisible()
  await chatInputA.fill(draft)
  await expect(chatInputA).toHaveValue(draft)

  // Tab B: switch variant to V3
  await tabB.goto("/settings/chat")
  const v3Card = tabB.getByRole("radio", { name: /split brief/i })
  await expect(v3Card).toBeVisible({ timeout: 15_000 })
  await v3Card.click()
  await expect(v3Card).toHaveAttribute("aria-checked", "true")

  // Tab A: variant should swap to V3 AND the draft should survive,
  // because the slot content (real ComposerTextarea) is the same node
  // shared between V1's textareaSlot and V3's textareaSlot.
  await expect(wrapperA.locator("[data-variant='v3']")).toBeVisible({
    timeout: 5_000,
  })
  const chatInputAfter = wrapperA.locator('[data-testid="chat-input"]')
  await expect(chatInputAfter).toBeVisible()
  await expect(chatInputAfter).toHaveValue(draft)

  await context.close()
})
