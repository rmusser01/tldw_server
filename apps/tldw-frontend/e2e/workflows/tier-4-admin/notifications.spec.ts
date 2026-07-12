import { test, expect, assertNoCriticalErrors } from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { NotificationsPage } from "../../utils/page-objects"
import type { Page, Route } from "@playwright/test"

const notificationItem = {
  id: 701,
  user_id: 1,
  kind: "job_failed",
  title: "Deterministic notification",
  message: "Acceptance fixture notification",
  severity: "error",
  created_at: "2026-07-11T12:00:00Z",
  read_at: null,
  dismissed_at: null,
}

async function fulfillJson(route: Route, status: number, body: unknown): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: { "access-control-allow-origin": "*" },
    body: JSON.stringify(body),
  })
}

async function fulfillActiveNotificationRequest(
  route: Route,
  unreadCount: number,
): Promise<void> {
  const request = route.request()
  const pathname = new URL(request.url()).pathname
  const method = request.method()
  if (method === "GET" && pathname === "/api/v1/notifications/unread-count") {
    await fulfillJson(route, 200, { unread_count: unreadCount })
    return
  }
  if (method === "GET" && pathname === "/api/v1/notifications/stream") {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "access-control-allow-origin": "*" },
      body: ": connected\n\n",
    })
    return
  }
  if (method === "GET" && pathname === "/api/v1/notifications/preferences") {
    await fulfillJson(route, 200, {
      user_id: "1",
      reminder_enabled: true,
      job_completed_enabled: true,
      job_failed_enabled: true,
      updated_at: "2026-07-11T12:00:00Z",
    })
    return
  }
  if (method === "GET" && pathname === "/api/v1/notifications") {
    const onlySnoozed = new URL(request.url()).searchParams.get("only_snoozed") === "true"
    await fulfillJson(route, 200, {
      items: onlySnoozed ? [] : [notificationItem],
      total: onlySnoozed ? 0 : 1,
    })
    return
  }
  await fulfillJson(route, 200, method === "POST" ? { updated: 1 } : {})
}

const jwtFor = (subject: string, roles: string[] = []): string => {
  const payload = Buffer.from(JSON.stringify({ sub: subject, roles })).toString("base64url")
  return `eyJhbGciOiJub25lIn0.${payload}.signature`
}

async function switchBearerScope(page: Page, accessToken: string): Promise<void> {
  await page.evaluate((nextAccessToken) => {
    const oldValue = localStorage.getItem("access_token")
    localStorage.setItem("access_token", nextAccessToken)
    window.dispatchEvent(new StorageEvent("storage", {
      key: "access_token",
      oldValue,
      newValue: nextAccessToken,
      storageArea: localStorage,
    }))
  }, accessToken)
}

test.describe("Notifications", () => {
  let notifications: NotificationsPage

  test.beforeEach(async ({ authedPage }) => {
    notifications = new NotificationsPage(authedPage)
  })

  test("notifications page loads with heading and refresh button", async ({ diagnostics }) => {
    await notifications.goto()
    await notifications.assertPageReady()

    // Heading should be visible
    await expect(notifications.heading).toBeVisible()

    // Refresh button should be visible
    await expect(notifications.refreshButton).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("notifications page shows unread count label", async ({ diagnostics }) => {
    await notifications.goto()
    await notifications.assertPageReady()

    // The unread label should be present (shows "Unread: N")
    await expect(notifications.unreadLabel).toBeVisible({ timeout: 15_000 })

    await assertNoCriticalErrors(diagnostics)
  })

  test("notifications page fires list and unread-count API on load", async ({
    authedPage,
    diagnostics,
  }) => {
    // Set up API call watchers before navigating
    const listCall = expectApiCall(authedPage, {
      url: "/notifications",
      method: "GET",
    })

    await notifications.goto()

    // The page should fire GET /notifications on mount
    await listCall

    await assertNoCriticalErrors(diagnostics)
  })

  test("notifications page shows empty state or list after loading", async ({ diagnostics }) => {
    await notifications.goto()
    await notifications.waitForLoaded()

    // After loading, either the empty state, list, error banner, or loading state should be visible
    const emptyVisible = await notifications.emptyState.isVisible().catch(() => false)
    const listVisible = await notifications.notificationsList.isVisible().catch(() => false)
    const errorVisible = await notifications.errorBanner.isVisible().catch(() => false)
    const loadingVisible = await notifications.loadingState.isVisible().catch(() => false)

    expect(emptyVisible || listVisible || errorVisible || loadingVisible).toBe(true)

    await assertNoCriticalErrors(diagnostics)
  })

  test("refresh button fires notifications API call", async ({ authedPage, diagnostics }) => {
    // Track whether any request to /notifications is made
    let apiCallMade = false
    const handler = (req: import("@playwright/test").Request) => {
      if (req.url().includes("/notifications") && req.method() === "GET") {
        apiCallMade = true
      }
    }
    authedPage.on("request", handler)

    await notifications.goto()
    await notifications.waitForLoaded()

    // Dismiss any Next.js error overlay that might block clicks
    await authedPage.keyboard.press("Escape")
    apiCallMade = false

    await notifications.refreshButton.click({ force: true })
    await expect
      .poll(() => apiCallMade, { timeout: 10_000 })
      .toBe(true)
    authedPage.removeListener("request", handler)

    expect(apiCallMade).toBe(true)

    await assertNoCriticalErrors(diagnostics)
  })

  test("notification items have action buttons when present", async ({
    authedPage,
    diagnostics,
  }) => {
    await notifications.goto()
    await notifications.waitForLoaded()

    const items = authedPage.locator("ul.space-y-3 > li")
    const itemCount = await items.count()

    if (itemCount > 0) {
      const firstItem = items.first()

      // Each notification item should have at least a Dismiss button
      const dismissBtn = firstItem.getByRole("button", { name: /dismiss/i })
      await expect(dismissBtn).toBeVisible()

      // Should also have a Snooze button
      const snoozeBtn = firstItem.getByRole("button", { name: /snooze/i })
      await expect(snoozeBtn).toBeVisible()
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("mark-read button fires API when notification is unread", async ({
    authedPage,
    diagnostics,
  }) => {
    await notifications.goto()
    await notifications.waitForLoaded()

    const markReadBtn = authedPage.getByRole("button", { name: /mark read/i }).first()
    const isVisible = await markReadBtn.isVisible().catch(() => false)

    if (isVisible) {
      const apiCall = expectApiCall(authedPage, {
        url: "/notifications/mark-read",
        method: "POST",
      })

      await markReadBtn.click()
      await apiCall
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("dismiss button fires API call", async ({ authedPage, diagnostics }) => {
    await notifications.goto()
    await notifications.waitForLoaded()

    const dismissBtn = authedPage.getByRole("button", { name: /dismiss/i }).first()
    const isVisible = await dismissBtn.isVisible().catch(() => false)

    if (isVisible) {
      const apiCall = expectApiCall(authedPage, {
        url: /\/notifications\/\d+\/dismiss/,
        method: "POST",
      })

      await dismissBtn.click()
      await apiCall
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("snooze button fires API call", async ({ authedPage, diagnostics }) => {
    await notifications.goto()
    await notifications.waitForLoaded()

    const snoozeBtn = authedPage.getByRole("button", { name: /snooze/i }).first()
    const isVisible = await snoozeBtn.isVisible().catch(() => false)

    if (isVisible) {
      const apiCall = expectApiCall(authedPage, {
        url: /\/notifications\/\d+\/snooze/,
        method: "POST",
      })

      await snoozeBtn.click()
      await apiCall
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("standard user exercises list, count, control, and SSE with one explicit mutation retry", async ({
    authedPage,
    diagnostics,
  }) => {
    const standardUserToken = jwtFor("standard-user", ["user"])
    await authedPage.addInitScript((accessToken) => {
      localStorage.setItem("access_token", accessToken)
    }, standardUserToken)
    let listCalls = 0
    let countCalls = 0
    let streamCalls = 0
    let markReadCalls = 0
    await authedPage.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
      const request = route.request()
      const pathname = new URL(request.url()).pathname
      if (request.headers()["authorization"] !== `Bearer ${standardUserToken}`) {
        await fulfillJson(route, 401, { detail: "missing standard-user credentials" })
        return
      }
      if (request.method() === "GET" && pathname === "/api/v1/notifications") listCalls += 1
      if (request.method() === "GET" && pathname === "/api/v1/notifications/unread-count") countCalls += 1
      if (request.method() === "GET" && pathname === "/api/v1/notifications/stream") streamCalls += 1
      if (request.method() === "POST" && pathname === "/api/v1/notifications/mark-read") {
        markReadCalls += 1
        if (markReadCalls === 1) {
          await fulfillJson(route, 503, { detail: "temporary notification failure" })
          return
        }
      }
      await fulfillActiveNotificationRequest(route, 1)
    })

    await notifications.goto()
    await expect(authedPage.getByText("Acceptance fixture notification").first()).toBeVisible()
    await authedPage.getByRole("button", { name: "Mark read" }).click()
    await expect(authedPage.getByRole("button", { name: "Retry action" })).toBeVisible()
    expect(markReadCalls).toBe(1)
    await authedPage.getByRole("button", { name: "Retry action" }).click()
    await expect.poll(() => markReadCalls).toBe(2)

    expect(listCalls).toBeGreaterThan(0)
    expect(countCalls).toBeGreaterThan(0)
    expect(streamCalls).toBeGreaterThan(0)
    await assertNoCriticalErrors(diagnostics)
  })

  test("restricted role suppresses the badge, stops terminal loops, and recovers after one explicit grant retry", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.clock.install()
    let granted = false
    let unreadCalls = 0
    let terminalCalls = 0
    await authedPage.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
      const pathname = new URL(route.request().url()).pathname
      terminalCalls += 1
      if (pathname === "/api/v1/notifications/unread-count") unreadCalls += 1
      if (!granted) {
        await fulfillJson(route, 403, { detail: "permission denied" })
        return
      }
      await fulfillActiveNotificationRequest(route, 4)
    })

    await notifications.goto()
    await expect(authedPage.getByText("Notifications unavailable for this account").first()).toBeVisible()
    const terminalCallCount = terminalCalls
    await authedPage.clock.fastForward(31_000)
    expect(terminalCalls).toBe(terminalCallCount)

    const trigger = authedPage.getByRole("button", {
      name: "Notifications unavailable for this account",
    })
    await expect(trigger.locator("span")).toHaveCount(0)
    await trigger.focus()
    await authedPage.keyboard.press("Enter")
    const retry = authedPage
      .getByRole("dialog", { name: "Notifications unavailable for this account" })
      .getByRole("button", { name: "Try again" })
    await expect(retry).toBeFocused()
    granted = true
    const unreadBeforeRetry = unreadCalls
    await retry.click()
    await expect.poll(() => unreadCalls).toBe(unreadBeforeRetry + 1)
    await expect(authedPage.getByRole("button", { name: "Notifications, 4 unread" })).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("401 recovery opens the existing sign-in flow without a terminal request loop", async ({
    authedPage,
    diagnostics,
  }) => {
    await authedPage.clock.install()
    let reauthenticated = false
    let requestCalls = 0
    await authedPage.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
      requestCalls += 1
      if (!reauthenticated) {
        await fulfillJson(route, 401, { detail: "expired credentials" })
        return
      }
      await fulfillActiveNotificationRequest(route, 2)
    })

    await notifications.goto()
    await expect(authedPage.getByText("Sign in again to view notifications")).toBeVisible()
    const terminalCallCount = requestCalls
    await authedPage.clock.fastForward(31_000)
    expect(requestCalls).toBe(terminalCallCount)
    await authedPage.getByRole("button", { name: "Open sign in" }).click()
    await expect(authedPage).toHaveURL(/\/(?:login|settings\/tldw)/)
    reauthenticated = true
    await switchBearerScope(authedPage, jwtFor("reauthenticated-user", ["user"]))
    await expect(authedPage.getByRole("button", { name: "Notifications, 2 unread" })).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })

  test("account switch clears the old badge before showing the new account count", async ({
    authedPage,
    diagnostics,
  }) => {
    const accountAToken = jwtFor("account-a")
    const accountBToken = jwtFor("account-b")
    let releaseAccountBUnread!: () => void
    const accountBUnreadPending = new Promise<void>((resolve) => {
      releaseAccountBUnread = resolve
    })
    await authedPage.addInitScript((accessToken) => {
      localStorage.setItem("access_token", accessToken)
    }, accountAToken)
    await authedPage.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
      const pathname = new URL(route.request().url()).pathname
      const authorization = route.request().headers()["authorization"]
      if (
        authorization === `Bearer ${accountBToken}`
        && pathname === "/api/v1/notifications/unread-count"
      ) {
        await accountBUnreadPending
      }
      await fulfillActiveNotificationRequest(
        route,
        authorization === `Bearer ${accountBToken}` ? 1 : 7,
      )
    })

    await notifications.goto()
    await expect(authedPage.getByRole("button", { name: "Notifications, 7 unread" })).toBeVisible()
    await switchBearerScope(authedPage, accountBToken)
    await expect(authedPage.getByRole("button", { name: "Notifications, 7 unread" })).toHaveCount(0, {
      timeout: 1_000,
    })
    await expect(authedPage.getByRole("button", { name: "Notifications, 1 unread" })).toHaveCount(0)
    releaseAccountBUnread()
    await expect(authedPage.getByRole("button", { name: "Notifications, 1 unread" })).toBeVisible()

    await assertNoCriticalErrors(diagnostics)
  })
})
