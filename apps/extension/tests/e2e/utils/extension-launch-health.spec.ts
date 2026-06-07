import { expect, test, type Page } from "@playwright/test"

import { launchWithBuiltExtension } from "./extension-build"

function watchLaunchFailures(page: Page) {
  const pageErrors: string[] = []
  const failedExtensionRequests: string[] = []

  page.on("pageerror", (error) => {
    pageErrors.push(error.message)
  })
  page.on("requestfailed", (request) => {
    const url = request.url()
    if (!url.startsWith("chrome-extension://")) return
    failedExtensionRequests.push(`${url} ${request.failure()?.errorText || ""}`.trim())
  })

  return {
    pageErrors,
    failedExtensionRequests,
  }
}

test.describe("Extension launch health", () => {
  test("opens packaged options route at Knowledge QA", async () => {
    test.setTimeout(120_000)
    test.fail(
      true,
      "TASK-2279.5 release blocker: packaged MV3 launch does not expose extension targets in headless mode and headed launch times out locally."
    )

    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {},
      allowOffline: false,
      launchTimeoutMs: 60_000,
    })
    const failures = watchLaunchFailures(page)

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })

      await expect(page).toHaveURL(/options\.html#\/knowledge$/)
      await expect(page.locator("#root")).not.toBeEmpty()
      await expect(page.getByTestId("knowledge-setup-diagnostics")).toBeVisible()
      await expect(page.getByRole("button", { name: "Finish setup" })).toBeVisible()

      expect(failures.pageErrors).toEqual([])
      expect(failures.failedExtensionRequests).toEqual([])
    } finally {
      await context.close()
    }
  })
})
