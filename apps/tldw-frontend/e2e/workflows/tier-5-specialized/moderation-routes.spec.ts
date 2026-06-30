/**
 * Moderation route E2E tests (Tier 5)
 *
 * Tests the canonical moderation routes:
 * - /moderation renders the review shell
 * - /moderation/rules renders content rule configuration
 * - /moderation-playground remains a legacy redirect
 *
 * Run: npx playwright test e2e/workflows/tier-5-specialized/moderation-routes.spec.ts
 */
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"

test.describe("Moderation routes", () => {
  test.beforeEach(async ({ authedPage }) => {
    await authedPage.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "ok" })
      })
    })

    await authedPage.route(/\/api\/v1\/moderation(?:\/.*)?$/, async (route) => {
      const url = new URL(route.request().url())
      const method = route.request().method()
      let body: unknown = { status: "ok" }

      if (url.pathname.endsWith("/settings")) {
        body = {
          pii_enabled: null,
          categories_enabled: null,
          effective: {
            pii_enabled: false,
            categories_enabled: []
          }
        }
      } else if (url.pathname.endsWith("/policy/effective")) {
        body = {
          enabled: false,
          input_action: "pass",
          output_action: "pass",
          categories_enabled: []
        }
      } else if (url.pathname.endsWith("/users")) {
        body = { overrides: {} }
      } else if (url.pathname.endsWith("/blocklist/managed")) {
        body = { version: "v1", items: [] }
      } else if (url.pathname.endsWith("/blocklist")) {
        body = method === "GET" ? [] : { status: "ok", count: 0 }
      } else if (url.pathname.endsWith("/test")) {
        body = {
          flagged: false,
          action: "pass",
          effective: {},
          category: null
        }
      }

      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body)
      })
    })
  })

  test("review route renders an honest queue shell", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/moderation", {
      waitUntil: "domcontentloaded",
    })

    await expect(
      authedPage.getByRole("heading", { name: /moderation review/i })
    ).toBeVisible({ timeout: 15_000 })
    await expect(authedPage.getByLabel(/status/i)).toBeVisible()
    await expect(authedPage.getByText(/review complete/i)).toBeVisible()
    await expect(authedPage.getByRole("link", { name: /^content rules$/i })).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("rules route loads content rule configuration", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/moderation/rules", {
      waitUntil: "domcontentloaded",
    })

    const heading = authedPage.getByRole("heading", {
      name: /content rules/i,
    })
    const permissionError = authedPage.getByText(
      /admin moderation access required/i
    )
    await expect(heading.or(permissionError).first()).toBeVisible({
      timeout: 15_000,
    })

    await assertNoCriticalErrors(diagnostics)
  })

  test("legacy playground route redirects to content rules", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/moderation-playground", {
      waitUntil: "domcontentloaded",
    })

    await expect(authedPage).toHaveURL(/\/moderation\/rules(?:$|[?#])/, {
      timeout: 15_000,
    })
    await expect(
      authedPage
        .getByRole("heading", { name: /content rules/i })
        .or(authedPage.getByText(/moderation playground has moved/i))
        .or(authedPage.getByText(/admin moderation access required/i))
        .first()
    ).toBeVisible({ timeout: 15_000 })

    await assertNoCriticalErrors(diagnostics)
  })

  test("policy settings fires API on rules load", async ({
    authedPage,
    diagnostics,
  }) => {
    const apiCall = expectApiCall(authedPage, {
      url: "/api/v1/moderation",
    })
    await authedPage.goto("/moderation/rules", {
      waitUntil: "domcontentloaded",
    })

    const { response } = await apiCall
    expect(response.status()).toBeLessThan(500)

    await assertNoCriticalErrors(diagnostics)
  })
})
