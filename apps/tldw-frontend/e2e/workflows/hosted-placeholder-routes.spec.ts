import { test, expect, assertNoCriticalErrors } from "../utils/fixtures"

const HOSTED_PLACEHOLDER_ROUTES = [
  {
    path: "/account",
    title: /Hosted Account Pages Live In The Private Distribution/i,
  },
  {
    path: "/billing",
    title: /Hosted Billing Lives In The Private Distribution/i,
  },
  {
    path: "/billing/success",
    title: /Hosted Billing Redirects Live In The Private Distribution/i,
  },
  {
    path: "/billing/cancel",
    title: /Hosted Billing Redirects Live In The Private Distribution/i,
  },
  {
    path: "/signup",
    title: /Signup Is Not Part Of The OSS Web Surface/i,
  },
  {
    path: "/auth/reset-password",
    title: /Password Reset Is Not Active Here/i,
  },
  {
    path: "/auth/magic-link",
    title: /Magic Link Sign-In Is Not Active Here/i,
  },
  {
    path: "/auth/verify-email",
    title: /Email Verification Is Not Active Here/i,
  },
]

const hostedMode =
  String(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || "").trim().toLowerCase() ===
  "hosted"

test.describe("Hosted placeholder routes", () => {
  for (const route of HOSTED_PLACEHOLDER_ROUTES) {
    test(`${route.path} renders its OSS placeholder`, async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.goto(route.path, { waitUntil: "domcontentloaded" })

      await expect(
        authedPage.getByTestId("route-placeholder-panel")
      ).toBeVisible({ timeout: 15_000 })
      await expect(
        authedPage.getByRole("heading", { name: route.title })
      ).toBeVisible({ timeout: 15_000 })

      await expect(authedPage.getByText(route.path, { exact: true })).toHaveCount(2)

      const primaryLink = authedPage.getByTestId("route-placeholder-primary")
      await expect(primaryLink).toHaveText(
        hostedMode ? "Open Login" : "Open Local Auth Settings"
      )
      await expect(primaryLink).toHaveAttribute(
        "href",
        hostedMode ? "/login" : "/settings/tldw"
      )

      await assertNoCriticalErrors(diagnostics)
    })
  }
})
