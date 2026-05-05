import { type Page, type Locator, expect } from "@playwright/test"
import { expectApiCall } from "../api-assertions"

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export type InteractiveExpectation =
  | { type: "api_call"; apiPattern: string | RegExp; method?: string }
  | { type: "modal"; modalSelector: string }
  | { type: "navigation"; targetUrl: string | RegExp }
  | { type: "state_change"; stateCheck: (page: Page) => Promise<unknown> }
  | { type: "toggle" }

export interface InteractiveElement {
  name: string
  locator: Locator
  expectation: InteractiveExpectation
  setup?: (page: Page) => Promise<void>
  cleanup?: (page: Page) => Promise<void>
}

/* ------------------------------------------------------------------ */
/* BasePage                                                            */
/* ------------------------------------------------------------------ */

export abstract class BasePage {
  constructor(protected page: Page) {}

  abstract goto(): Promise<void>

  abstract assertPageReady(): Promise<void>

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return []
  }

  async assertAllButtonsWired(): Promise<void> {
    const elements = await this.getInteractiveElements()
    if (elements.length === 0) {
      throw new Error(
        `${this.constructor.name}.getInteractiveElements() returned empty array. ` +
        `Override it to declare interactive elements, or remove assertAllButtonsWired() call.`
      )
    }

    for (const el of elements) {
      const visible = await el.locator.isVisible().catch(() => false)
      if (!visible) continue

      const enabled = await el.locator.isEnabled().catch(() => false)
      if (!enabled) continue

      if (el.setup) await el.setup(this.page)

      try {
        switch (el.expectation.type) {
          case "api_call": {
            const call = expectApiCall(this.page, {
              url: el.expectation.apiPattern,
              method: el.expectation.method,
            }, 10_000)
            await el.locator.click()
            try {
              await call
            } catch (err) {
              throw new Error(
                `Button "${el.name}" expected to fire API call ` +
                `${el.expectation.method ?? "ANY"} ${el.expectation.apiPattern} ` +
                `but no matching call was made. Original: ${(err as Error).message}`
              )
            }
            break
          }

          case "modal": {
            await el.locator.click()
            const modal = this.page.locator(el.expectation.modalSelector)
            await expect(modal).toBeVisible({ timeout: 5_000 })
            await this.page.keyboard.press("Escape")
            await expect(modal).toBeHidden({ timeout: 3_000 }).catch(() => {})
            break
          }

          case "navigation": {
            const targetUrl = el.expectation.targetUrl
            await el.locator.click()
            if (typeof targetUrl === "string") {
              await expect(this.page).toHaveURL(new RegExp(targetUrl), { timeout: 10_000 })
            } else {
              await expect(this.page).toHaveURL(targetUrl, { timeout: 10_000 })
            }
            await this.page.goBack()
            await this.assertPageReady()
            break
          }

          case "state_change": {
            const stateCheck = el.expectation.stateCheck
            const before = await stateCheck(this.page)
            await el.locator.click()
            const beforeSerialized = JSON.stringify(before)
            try {
              await expect
                .poll(
                  async () => JSON.stringify(await stateCheck(this.page)),
                  { timeout: 5_000 }
                )
                .not.toBe(beforeSerialized)
            } catch {
              throw new Error(
                `Button "${el.name}" expected state change but state is identical before and after click.`
              )
            }
            break
          }

          case "toggle": {
            const beforeChecked = await el.locator.isChecked().catch(() => null)
            await el.locator.click()
            if (beforeChecked !== null) {
              const afterChecked = await el.locator.isChecked().catch(() => null)
              if (beforeChecked === afterChecked) {
                throw new Error(`Toggle "${el.name}" did not change checked state after click.`)
              }
            }
            break
          }
        }
      } finally {
        if (el.cleanup) await el.cleanup(this.page).catch(() => {})
      }
    }
  }
}
