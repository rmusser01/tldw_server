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

const engineReviewItem = {
  ...baseReviewItem,
  phase: "unspecified",
  source_type: "moderation_engine",
  category: "moderation_engine_failure",
  excerpt: "[Moderation execution failed; content withheld]",
  context: { reason: "deadline_exceeded", operation: "evaluate", correlation_id: "1234567890abcdef1234567890abcdef" },
  effective_policy: {},
  matches: [],
  recommended_action: "escalate",
}

const fulfillJson = async (route: Route, body: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  })
}

const mockModerationReviewApi = async (page: Page, seed = baseReviewItem as typeof baseReviewItem | typeof engineReviewItem) => {
  let currentItem = { ...seed }

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
        ...seed,
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
      currentItem = { ...seed }
      await fulfillJson(route, currentItem)
      return
    }

    if (url.pathname === "/api/v1/moderation/review/bulk-decision") {
      currentItem = { ...seed, status: "dismissed" }
      await fulfillJson(route, {
        results: [{ item_id: seed.id, ok: true, item: currentItem }],
        ok_count: 1, error_count: 0,
      })
      return
    }

    if (url.pathname === "/api/v1/moderation/review/audit") {
      await fulfillJson(route, { events: [], next_cursor: null })
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

  test("reviews one neutral operational incident without changing its identity", async ({ authedPage, diagnostics }) => {
    await mockModerationReviewApi(authedPage, engineReviewItem)
    await authedPage.goto("/moderation", { waitUntil: "domcontentloaded" })
    await expect(authedPage.getByRole("heading", { name: "Moderation execution failure" })).toBeVisible()
    await expect(authedPage.getByText("Unspecified").first()).toBeVisible()
    await expect(authedPage.getByText("deadline_exceeded")).toBeVisible()
    await expect(authedPage.getByText("Matches", { exact: true })).toHaveCount(0)
    await expect(authedPage.getByRole("listitem")).toHaveCount(1)
    await authedPage.getByRole("button", { name: /^approve$/i }).click()
    await authedPage.getByRole("button", { name: /undo decision/i }).click()
    await authedPage.getByRole("checkbox", { name: "Select review item rev-1" }).check()
    const bulk = authedPage.waitForRequest((request) => request.url().endsWith("/review/bulk-decision"))
    await authedPage.getByRole("button", { name: "Dismiss selected" }).click()
    expect((await bulk).postDataJSON().item_ids).toEqual(["rev-1"])
    await authedPage.getByRole("combobox", { name: "Status", exact: true }).selectOption("")
    await authedPage.getByRole("button", { name: "Refresh", exact: true }).click()
    await expect(authedPage.getByRole("listitem")).toHaveCount(1)
    await expect(authedPage.getByText("Unspecified").first()).toBeVisible()
    const excerptWidth = await authedPage.getByRole("listitem").getByText(engineReviewItem.excerpt, { exact: true }).evaluate((element) => element.clientWidth)
    expect(excerptWidth).toBeGreaterThanOrEqual(120)
    await expectNoPageHorizontalOverflow(authedPage, "desktop execution incident")
    await authedPage.screenshot({ path: "/tmp/task3-incident-desktop.png", fullPage: true })
    await authedPage.setViewportSize({ width: 390, height: 900 })
    await expectNoPageHorizontalOverflow(authedPage, "mobile execution incident")
    await authedPage.screenshot({ path: "/tmp/task3-incident-mobile.png", fullPage: true })
    await authedPage.setViewportSize({ width: 1920, height: 1080 })
    await expectNoPageHorizontalOverflow(authedPage, "wide execution incident")
    const wideExcerptWidth = await authedPage.getByRole("listitem").getByText(engineReviewItem.excerpt, { exact: true }).evaluate((element) => element.clientWidth)
    expect(wideExcerptWidth).toBeGreaterThanOrEqual(120)
    await authedPage.screenshot({ path: "/tmp/task3-incident-wide.png", fullPage: true })
    await authedPage.setViewportSize({ width: 1024, height: 900 })
    await expect(authedPage.getByText("Excerpt", { exact: true })).toBeVisible()
    await expectNoPageHorizontalOverflow(authedPage, "table execution incident")
    const tableExcerptWidth = await authedPage.getByRole("listitem").getByText(engineReviewItem.excerpt, { exact: true }).evaluate((element) => element.clientWidth)
    expect(tableExcerptWidth).toBeGreaterThanOrEqual(120)
    await authedPage.screenshot({ path: "/tmp/task3-incident-tablet.png", fullPage: true })
    await assertNoCriticalErrors(diagnostics)
  })
})
