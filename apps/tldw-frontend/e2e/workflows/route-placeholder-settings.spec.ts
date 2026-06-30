import { test, expect, assertNoCriticalErrors } from "../utils/fixtures"

const SETTINGS_PLACEHOLDER_ROUTES = [
  {
    path: "/connectors",
    title: /^Connectors$/i,
  },
  {
    path: "/config",
    title: /Configuration Center Is Coming Soon/i,
  },
  {
    path: "/profile",
    title: /Profile Page Is Coming Soon/i,
  },
]

test.describe("RoutePlaceholder settings CTA contract", () => {
  for (const route of SETTINGS_PLACEHOLDER_ROUTES) {
    test(`${route.path} renders a single settings CTA`, async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.goto(route.path, { waitUntil: "domcontentloaded" })

      await expect(
        authedPage.getByRole("heading", { name: route.title })
      ).toBeVisible({ timeout: 15_000 })

      const settingsLinks = authedPage.getByRole("link", {
        name: /^Open Settings$/i,
      })
      await expect(settingsLinks).toHaveCount(1)
      await expect(settingsLinks.first()).toHaveAttribute("href", "/settings")
      await expect(
        authedPage.getByTestId("route-placeholder-open-settings")
      ).toHaveCount(0)

      await assertNoCriticalErrors(diagnostics)
    })
  }
})
