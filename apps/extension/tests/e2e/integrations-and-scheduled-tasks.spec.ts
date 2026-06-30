import { expect, test } from "@playwright/test"
import { launchWithBuiltExtensionOrSkip } from "./utils/real-server"

const goToOptionsRoute = async (page: any, optionsUrl: string, route: string) => {
  await page.goto(`${optionsUrl}#${route}`, { waitUntil: "domcontentloaded" })
}

test.describe("Integrations and scheduled tasks routes", () => {
  test("renders the new management surfaces in the extension options UI", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtensionOrSkip(test, {
      seedConfig: {
        __tldw_first_run_complete: true,
        __tldw_allow_offline: true
      }
    })

    try {
      await goToOptionsRoute(page, optionsUrl, "/integrations")
      await expect(
        page.getByRole("heading", { name: /personal integrations/i })
      ).toBeVisible()

      await goToOptionsRoute(page, optionsUrl, "/admin/integrations")
      await expect(
        page.getByRole("heading", { name: /workspace integrations/i })
      ).toBeVisible()

      await goToOptionsRoute(page, optionsUrl, "/scheduled-tasks")
      await expect(
        page.getByRole("heading", { name: /scheduled tasks/i })
      ).toBeVisible()
      await expect(
        page.getByText(/Track reminders, Watchlist monitors/i)
      ).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
