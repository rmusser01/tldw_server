/**
 * Moderation review power-user E2E checks (Tier 5)
 *
 * Run: bunx playwright test e2e/workflows/tier-5-specialized/moderation-review-power-user.spec.ts --project=tier-5
 */
import {
  test,
  expect,
  assertNoCriticalErrors,
  loadModerationReviewItemsFixture,
} from "../../utils/fixtures"
import type { Page, Route } from "@playwright/test"

const reviewItems = loadModerationReviewItemsFixture().populated

const fulfillJson = async (route: Route, body: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  })
}

const mockModerationReviewApi = async (page: Page) => {
  let currentItems = reviewItems.map((item) => ({ ...item }))

  await page.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
    await fulfillJson(route, { status: "ok" })
  })

  await page.route(/\/api\/v1\/moderation(?:\/.*)?$/, async (route) => {
    const url = new URL(route.request().url())

    if (url.pathname === "/api/v1/moderation/review/items") {
      const requestedStatus = url.searchParams.get("status")
      const items = currentItems.filter((item) => !requestedStatus || item.status === requestedStatus)
      await fulfillJson(route, { items, total: items.length, next_cursor: null })
      return
    }

    const detailMatch = url.pathname.match(/\/api\/v1\/moderation\/review\/items\/([^/]+)$/)
    if (detailMatch) {
      const item = currentItems.find((entry) => entry.id === detailMatch[1])
      await fulfillJson(route, item || { detail: "not found" }, item ? 200 : 404)
      return
    }

    if (url.pathname === "/api/v1/moderation/review/bulk-decision") {
      const body = JSON.parse(route.request().postData() || "{}")
      const requestedIds: string[] = Array.isArray(body.item_ids) ? body.item_ids : []
      currentItems = currentItems.map((item) =>
        item.id === "rev-1" ? { ...item, status: "dismissed", updated_at: "2026-05-12T23:15:00.000Z" } : item
      )
      await fulfillJson(route, {
        ok_count: requestedIds.includes("rev-1") ? 1 : 0,
        error_count: requestedIds.includes("rev-2") ? 1 : 0,
        results: requestedIds.map((itemId) =>
          itemId === "rev-2"
            ? {
                item_id: itemId,
                ok: false,
                error: "not_found",
              }
            : {
                item_id: itemId,
                ok: true,
                item: currentItems.find((entry) => entry.id === itemId),
                decision: {
                  id: `decision-${itemId}`,
                  item_id: itemId,
                  action: "dismiss",
                  status: "dismissed",
                  previous_status: "needs_review",
                  decided_by: "moderator-local",
                  decided_at: "2026-05-12T23:15:00.000Z",
                  undo_expires_at: "2026-05-12T23:30:00.000Z",
                  undo_token: "bulk-undo-token",
                },
              }
        ),
      })
      return
    }

    await fulfillJson(route, { status: "ok" })
  })
}

test.describe("Moderation review power-user workflow", () => {
  test.beforeEach(async ({ authedPage }) => {
    await mockModerationReviewApi(authedPage)
  })

  test("bulk dismisses selected items and reports partial failures", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.goto("/moderation", { waitUntil: "domcontentloaded" })

    await expect(
      authedPage.getByRole("heading", { name: /moderation review/i })
    ).toBeVisible({ timeout: 15_000 })
    await expect(authedPage.getByText(reviewItems[0].excerpt).first()).toBeVisible()

    await authedPage.getByLabel(/select review item rev-1/i).check()
    await authedPage.getByLabel(/select review item rev-2/i).check()
    await expect(authedPage.getByText(/2 selected/i)).toBeVisible()

    await authedPage.getByRole("button", { name: /dismiss selected/i }).click()
    await expect(authedPage.getByText(/1 updated/i)).toBeVisible()
    await expect(authedPage.getByText(/1 failed/i)).toBeVisible()
    await expect(authedPage.getByText(/rev-2: not_found/i)).toBeVisible()

    await authedPage.getByRole("button", { name: /clear selection/i }).click()
    await expect(authedPage.getByTestId("moderation-bulk-decision-bar")).toHaveCount(0)

    await assertNoCriticalErrors(diagnostics)
  })
})
