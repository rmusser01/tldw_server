import { expect, test } from "@playwright/test"
import { dismissQuickIngest } from "./utils/journey-helpers"

test.describe("Quick Ingest terminal close", () => {
  test("uses Done without sending Escape", async ({ page }) => {
    await page.setContent(`
      <div role="dialog" aria-label="Quick Ingest">
        <button
          type="button"
          onclick="this.closest('[role=dialog]').setAttribute('hidden', '')"
        >
          Done
        </button>
      </div>
      <script>
        window.terminalCloseEscapeSeen = false
        window.addEventListener("keydown", (event) => {
          if (event.key === "Escape") window.terminalCloseEscapeSeen = true
        })
      </script>
    `)

    await dismissQuickIngest(page, { terminal: true })

    await expect(page.getByRole("dialog", { name: "Quick Ingest" })).toBeHidden()
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (window as typeof window & { terminalCloseEscapeSeen?: boolean })
              .terminalCloseEscapeSeen ?? false
        )
      )
      .toBe(false)
  })

  test("refuses Escape when terminal controls are unavailable", async ({ page }) => {
    await page.setContent(`
      <div role="dialog" aria-label="Quick Ingest">
        <p>Terminal results</p>
      </div>
      <script>
        window.terminalCloseEscapeSeen = false
        window.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            window.terminalCloseEscapeSeen = true
            document.querySelector('[role=dialog]').setAttribute('hidden', '')
          }
        })
      </script>
    `)

    await expect(dismissQuickIngest(page, { terminal: true })).rejects.toThrow(
      "Quick Ingest terminal dialog is missing a visible Done or modal close control"
    )
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (window as typeof window & { terminalCloseEscapeSeen?: boolean })
              .terminalCloseEscapeSeen ?? false
        )
      )
      .toBe(false)
  })
})
