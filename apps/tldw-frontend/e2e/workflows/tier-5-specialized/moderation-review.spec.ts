/**
 * Moderation review queue E2E checks (Tier 5)
 *
 * Run: bunx playwright test e2e/workflows/tier-5-specialized/moderation-review.spec.ts --project=tier-5
 */
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import type { Page, Route } from "@playwright/test"

const baseReviewItem = {
  id: "rev-1",
  status: "needs_review",
  phase: "input",
  source_type: "chat",
  source_id: "conversation-42",
  user_id: "user-alex",
  session_id: "session-7",
  created_at: "2026-05-12T22:52:00.000Z",
  updated_at: "2026-05-12T22:52:00.000Z",
  severity: "high",
  category: "pii",
  safe_fields: {
    excerpt: true,
    context: true,
    effective_policy: true,
    matches: true,
  },
  excerpt: "Possible credential leak in chat input: sk-...redacted",
  context: {
    route: "/chat",
    model: "local-llm",
    capture: "sanitized",
  },
  effective_policy: {
    pii_enabled: true,
    input_action: "block",
    output_action: "redact",
  },
  matches: [
    {
      rule_id: "pii-api-key",
      pattern_type: "pii",
      category: "credential",
      action: "block",
      sample: "sk-...redacted",
      confidence: 0.94,
    },
  ],
  recommended_action: "block",
  retention_expires_at: "2026-06-11T22:52:00.000Z",
  content_redacted_at: null,
}

const fulfillJson = async (route: Route, body: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  })
}

const mockModerationReviewApi = async (page: Page) => {
  let currentItem = { ...baseReviewItem }

  await page.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
    await fulfillJson(route, { status: "ok" })
  })

  await page.route(/\/api\/v1\/moderation(?:\/.*)?$/, async (route) => {
    const url = new URL(route.request().url())
    const method = route.request().method()

    if (url.pathname === "/api/v1/moderation/review/items") {
      const requestedStatus = url.searchParams.get("status")
      const items = !requestedStatus || currentItem.status === requestedStatus ? [currentItem] : []
      await fulfillJson(route, { items, total: items.length, next_cursor: null })
      return
    }

    if (url.pathname === "/api/v1/moderation/review/items/rev-1") {
      await fulfillJson(route, currentItem)
      return
    }

    if (url.pathname === "/api/v1/moderation/review/items/rev-1/decision") {
      currentItem = {
        ...baseReviewItem,
        status: "approved",
        updated_at: "2026-05-12T23:15:00.000Z",
      }
      await fulfillJson(route, {
        item: currentItem,
        undo_token: "undo-token-1",
        decision: {
          id: "decision-1",
          item_id: "rev-1",
          action: "approve",
          status: "approved",
          previous_status: "needs_review",
          decided_by: "moderator-local",
          reason: null,
          decided_at: "2026-05-12T23:15:00.000Z",
          undo_token: "undo-token-1",
        },
      })
      return
    }

    if (url.pathname === "/api/v1/moderation/review/items/rev-1/undo") {
      currentItem = { ...baseReviewItem }
      await fulfillJson(route, currentItem)
      return
    }

    await fulfillJson(route, method === "GET" ? { status: "ok" } : { status: "ok" })
  })
}

const expectNoPageHorizontalOverflow = async (page: Page, label: string) => {
  const overflow = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(
    overflow.scrollWidth,
    `${label} overflowed viewport: ${JSON.stringify(overflow)}`
  ).toBeLessThanOrEqual(overflow.clientWidth + 1)
}

test.describe("Moderation review queue", () => {
  test.beforeEach(async ({ authedPage }) => {
    await mockModerationReviewApi(authedPage)
  })

  test("supports list detail decision and undo affordance", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/moderation", { waitUntil: "domcontentloaded" })

    await expect(
      authedPage.getByRole("heading", { name: /moderation review/i })
    ).toBeVisible({ timeout: 15_000 })
    await expect(authedPage.getByText(baseReviewItem.excerpt).first()).toBeVisible()
    await expect(authedPage.getByText(/rule: pii-api-key/i)).toBeVisible()
    await expect(authedPage.getByText(/94% confidence/i)).toBeVisible()
    await expect(authedPage.getByRole("button", { name: /^approve$/i })).toBeVisible()
    await expect(authedPage.getByRole("button", { name: /^escalate$/i })).toBeVisible()

    await authedPage.getByRole("button", { name: /^approve$/i }).click()
    await expect(authedPage.getByRole("button", { name: /undo decision/i })).toBeVisible()
    await expect(authedPage.getByText(/approved/i).first()).toBeVisible()
    await expect(
      authedPage.getByText(/active filters no longer include this selected item/i)
    ).toBeVisible()
    await expectNoPageHorizontalOverflow(authedPage, "desktop moderation review")

    await assertNoCriticalErrors(diagnostics)
  })

  test("keeps the review queue usable at mobile width", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.setViewportSize({ width: 390, height: 900 })
    await authedPage.goto("/moderation", { waitUntil: "domcontentloaded" })

    await expect(
      authedPage.getByRole("heading", { name: /moderation review/i })
    ).toBeVisible({ timeout: 15_000 })
    await expect(authedPage.getByLabel(/status/i)).toBeVisible()
    await expect(authedPage.getByText(baseReviewItem.excerpt).first()).toBeVisible()
    await expect(authedPage.getByRole("button", { name: /^approve$/i })).toBeVisible()
    await expectNoPageHorizontalOverflow(authedPage, "mobile moderation review")

    await assertNoCriticalErrors(diagnostics)
  })
})
