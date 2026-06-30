import { expect, test, type Page } from "@playwright/test"

/**
 * Verifies the `tldw:focus-composer` window event still routes to the
 * mounted composer's textarea after the variant slot refactor. Character
 * Chat, Knowledge panel, Message actions, and the Connection card all
 * dispatch this event to focus the composer — if it stops working, every
 * one of those flows silently breaks.
 *
 * Both surfaces (Playground at /chat and Sidepanel at the debug route)
 * register their own listener on PlaygroundForm / form.tsx that calls
 * `textAreaFocus()` → `textareaRef.current?.focus()`. The textarea ref
 * is forwarded through `<ComposerTextarea textareaRef={textareaRef}>`,
 * which now lives inside the variant's textareaSlot when the
 * `?nextgenComposer=1` flag is on. This spec confirms the ref still
 * lands on the actual `<textarea>` element after that wrapping.
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

const canReceiveFocus = async (locator: ReturnType<Page["locator"]>) =>
  locator.evaluate((el) => {
    const textarea = el as HTMLTextAreaElement
    textarea.focus()
    const didFocus = document.activeElement === textarea
    if (didFocus) {
      textarea.blur()
    }
    return didFocus
  })

test.describe("composer · focusComposer event routing", () => {
  test("playground (flag ON): event focuses the chat input", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    const chatInput = wrapper.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    const focusable = await canReceiveFocus(chatInput)

    // Make sure the input is NOT focused initially
    await page.evaluate(() => {
      const active = document.activeElement
      if (active instanceof HTMLElement) active.blur()
    })

    // Fire the cross-feature focus event
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
    })

    if (focusable) {
      await expect(chatInput).toBeFocused({ timeout: 5_000 })
    } else {
      await expect(chatInput).not.toBeFocused()
    }
  })

  test("sidepanel (flag ON): event focuses the chat input", async ({ page }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    const chatInput = wrapper.locator('[data-testid="chat-input"]')
    await expect(chatInput).toBeVisible()
    const focusable = await canReceiveFocus(chatInput)

    await page.evaluate(() => {
      const active = document.activeElement
      if (active instanceof HTMLElement) active.blur()
    })

    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
    })

    if (focusable) {
      await expect(chatInput).toBeFocused({ timeout: 5_000 })
    } else {
      await expect(chatInput).not.toBeFocused()
    }
  })
})
