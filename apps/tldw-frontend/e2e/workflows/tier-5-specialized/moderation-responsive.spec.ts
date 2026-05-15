/**
 * Moderation responsive checks (Tier 5)
 *
 * Run: bunx playwright test e2e/workflows/tier-5-specialized/moderation-responsive.spec.ts --project=tier-5
 */
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { waitForVisualSettle } from "../../utils/helpers"

const mockModerationApi = async (page: import("@playwright/test").Page) => {
  await page.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" })
    })
  })

  await page.route(/\/api\/v1\/moderation(?:\/.*)?$/, async (route) => {
    const url = new URL(route.request().url())
    const method = route.request().method()
    let body: unknown = { status: "ok" }

    if (url.pathname.endsWith("/settings")) {
      body = {
        pii_enabled: true,
        categories_enabled: ["pii", "violence"],
        effective: {
          pii_enabled: true,
          categories_enabled: ["pii", "violence"]
        }
      }
    } else if (url.pathname.endsWith("/policy/effective")) {
      body = {
        enabled: true,
        input_enabled: true,
        output_enabled: true,
        input_action: "block",
        output_action: "redact",
        redact_replacement: "[REDACTED]",
        blocklist_count: 14,
        categories_enabled: ["pii", "violence"]
      }
    } else if (url.pathname.endsWith("/users")) {
      body = {
        overrides: {
          "alpha-user-with-long-id@example.test": {
            enabled: true,
            input_action: "block",
            output_action: "warn",
            rules: [{ id: "r1", pattern: "private-token", action: "block", phase: "both" }]
          },
          "beta-user-with-an-even-longer-identifier@example.test": {
            enabled: true,
            input_action: "warn",
            output_action: "redact"
          }
        }
      }
    } else if (url.pathname.endsWith("/blocklist/managed")) {
      body = {
        version: "responsive-v1",
        items: Array.from({ length: 14 }, (_, index) => ({
          id: index + 1,
          line: `very-long-sensitive-pattern-${index + 1}-with-extra-context -> ${index % 2 === 0 ? "block" : "redact"} #pii,violence`,
          pattern_type: "literal",
          action: index % 2 === 0 ? "block" : "redact",
          categories: ["pii", "violence"],
          ok: true
        }))
      }
    } else if (url.pathname.endsWith("/blocklist")) {
      body = method === "GET"
        ? ["private-token -> block #pii", "confidential project codename -> redact:[REDACTED] #secrets"]
        : { status: "ok", count: 2 }
    } else if (url.pathname.endsWith("/test")) {
      body = {
        flagged: true,
        action: "block",
        sample: "private-token",
        effective: { enabled: true },
        category: "pii"
      }
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body)
    })
  })
}

const expectNoPageHorizontalOverflow = async (page: import("@playwright/test").Page, label: string) => {
  const overflow = await page.evaluate(() => {
    const root = document.documentElement
    const offenders = Array.from(document.querySelectorAll<HTMLElement>("body *"))
      .filter((element) => {
        const rect = element.getBoundingClientRect()
        return rect.width > 0 && rect.right > root.clientWidth + 1
      })
      .slice(0, 5)
      .map((element) => ({
        tag: element.tagName.toLowerCase(),
        className: element.className,
        testId: element.getAttribute("data-testid"),
        role: element.getAttribute("role"),
        text: (element.textContent || "").trim().slice(0, 80),
        right: Math.round(element.getBoundingClientRect().right),
        width: Math.round(element.getBoundingClientRect().width)
      }))

    return {
      clientWidth: root.clientWidth,
      scrollWidth: root.scrollWidth,
      offenders
    }
  })

  expect(
    overflow.scrollWidth,
    `${label} overflowed: ${JSON.stringify(overflow.offenders)}`
  ).toBeLessThanOrEqual(overflow.clientWidth + 1)
}

test.describe("Moderation rules responsive layout", () => {
  test.beforeEach(async ({ authedPage }) => {
    await authedPage.addInitScript(() => {
      localStorage.setItem("moderation-playground-onboarded", "true")
    })
    await mockModerationApi(authedPage)
  })

  for (const viewport of [
    { width: 390, height: 844, label: "mobile" },
    { width: 768, height: 1024, label: "tablet" },
    { width: 1440, height: 900, label: "desktop" }
  ]) {
    test(`keeps rule configuration inside the viewport at ${viewport.label}`, async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.setViewportSize({ width: viewport.width, height: viewport.height })
      await authedPage.goto("/moderation/rules", { waitUntil: "domcontentloaded" })
      await waitForVisualSettle(authedPage)

      await expect(authedPage.getByRole("heading", { name: /content rules/i })).toBeVisible({
        timeout: 15_000
      })
      await expectNoPageHorizontalOverflow(authedPage, `${viewport.label} policy tab`)

      await authedPage.getByRole("tab", { name: /blocklist studio/i }).click()
      await expect(authedPage.getByTestId("rules-table")).toBeVisible({ timeout: 15_000 })
      await expectNoPageHorizontalOverflow(authedPage, `${viewport.label} blocklist tab`)

      await authedPage.getByRole("tab", { name: /user overrides/i }).click()
      await expect(authedPage.getByTestId("overrides-table")).toBeVisible({ timeout: 15_000 })
      await expectNoPageHorizontalOverflow(authedPage, `${viewport.label} overrides tab`)

      await authedPage.getByRole("tab", { name: /test sandbox/i }).click()
      await expect(authedPage.getByRole("button", { name: /run test/i })).toBeVisible()
      await expectNoPageHorizontalOverflow(authedPage, `${viewport.label} test tab`)

      await assertNoCriticalErrors(diagnostics)
    })
  }
})
