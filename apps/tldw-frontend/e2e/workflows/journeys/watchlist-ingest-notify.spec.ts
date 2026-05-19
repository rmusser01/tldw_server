import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { waitForConnection } from "../../utils/helpers"
import { NotificationsPage } from "../../utils/page-objects"

type MockWatchlistSource = {
  id: number
  name: string
  url: string
  source_type: "rss"
  active: boolean
  tags: string[]
  created_at: string
  updated_at: string
}

type MockWatchlistJob = {
  id: number
  name: string
  description: string
  scope: { sources: number[] }
  schedule_expr: string
  timezone: string
  active: boolean
  created_at: string
  updated_at: string
  last_run_at: string | null
  next_run_at: string | null
}

type MockWatchlistRun = {
  id: number
  job_id: number
  status: "running" | "completed"
  started_at: string
  finished_at: string | null
  stats: {
    items_found: number
    items_ingested: number
  }
}

type MockScrapedItem = {
  id: number
  run_id: number
  job_id: number
  source_id: number
  media_id: number | null
  media_uuid: string | null
  url: string
  title: string
  summary: string
  content: string
  published_at: string
  tags: string[]
  status: "ingested"
  reviewed: boolean
  queued_for_briefing: boolean
  created_at: string
}

type MockNotification = {
  id: number
  kind: string
  title: string
  message: string
  severity: "info"
  created_at: string
  read_at: string | null
  dismissed_at: string | null
}

const SOURCE_ID = 101
const JOB_ID = 300
const RUN_ID = 7001
const ITEM_ID = 9001
const NOTIFICATION_ID = 8801

const WATCHLIST_SOURCE: MockWatchlistSource = {
  id: SOURCE_ID,
  name: "Morning Feed",
  url: "https://example.com/feed.xml",
  source_type: "rss",
  active: true,
  tags: ["news"],
  created_at: "2026-03-20T08:00:00Z",
  updated_at: "2026-03-20T08:00:00Z",
}

const WATCHLIST_JOB: MockWatchlistJob = {
  id: JOB_ID,
  name: "Morning Brief",
  description: "Tracks the morning feed and persists new items.",
  scope: { sources: [SOURCE_ID] },
  schedule_expr: "0 8 * * *",
  timezone: "UTC",
  active: true,
  created_at: "2026-03-20T08:00:00Z",
  updated_at: "2026-03-20T08:00:00Z",
  last_run_at: null,
  next_run_at: "2026-03-22T08:00:00Z",
}

const buildIngestedItem = (): MockScrapedItem => ({
  id: ITEM_ID,
  run_id: RUN_ID,
  job_id: JOB_ID,
  source_id: SOURCE_ID,
  media_id: null,
  media_uuid: null,
  url: "https://example.com/story/morning-brief",
  title: "Morning Brief catches a new article",
  summary: "The feed picked up a fresh article during the manual run.",
  content:
    "<p>The manual watchlist run fetched one new article and stored it for review.</p>",
  published_at: "2026-03-21T07:45:00Z",
  tags: ["news"],
  status: "ingested",
  reviewed: false,
  queued_for_briefing: false,
  created_at: "2026-03-21T07:46:00Z",
})

const buildCompletedRun = (): MockWatchlistRun => ({
  id: RUN_ID,
  job_id: JOB_ID,
  status: "completed",
  started_at: "2026-03-21T07:45:00Z",
  finished_at: "2026-03-21T07:46:00Z",
  stats: {
    items_found: 1,
    items_ingested: 1,
  },
})

const buildTriggeredRun = (): MockWatchlistRun => ({
  ...buildCompletedRun(),
  status: "running",
  finished_at: null,
})

const buildNotification = (): MockNotification => ({
  id: NOTIFICATION_ID,
  kind: "watchlist_run",
  title: "Run completed",
  message: "Morning Brief ingested 1 new article.",
  severity: "info",
  created_at: "2026-03-21T07:46:30Z",
  read_at: null,
  dismissed_at: null,
})

const jsonResponse = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

const setupWatchlistJourneyRoutes = async (page: Page) => {
  const state = {
    jobs: [{ ...WATCHLIST_JOB }],
    runs: [] as MockWatchlistRun[],
    items: [] as MockScrapedItem[],
    notifications: [] as MockNotification[],
  }

  await page.route(/\/api\/v1\/watchlists(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const { pathname, searchParams } = url
    const pageNum = Number(searchParams.get("page") || "1")
    const size = Number(searchParams.get("size") || "25")

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/sources") {
      await jsonResponse(route, {
        items: [WATCHLIST_SOURCE],
        total: 1,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/jobs") {
      await jsonResponse(route, {
        items: state.jobs,
        total: state.jobs.length,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "POST" && pathname === `/api/v1/watchlists/jobs/${JOB_ID}/run`) {
      state.jobs = state.jobs.map((job) =>
        job.id === JOB_ID
          ? { ...job, last_run_at: "2026-03-21T07:45:00Z" }
          : job
      )
      state.runs = [buildCompletedRun()]
      state.items = [buildIngestedItem()]
      state.notifications = [buildNotification()]

      await jsonResponse(route, buildTriggeredRun())
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/runs") {
      await jsonResponse(route, {
        items: state.runs,
        total: state.runs.length,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/items/smart-counts") {
      await jsonResponse(route, {
        all: state.items.length,
        today: state.items.length,
        today_unread: state.items.filter((item) => !item.reviewed).length,
        unread: state.items.filter((item) => !item.reviewed).length,
        reviewed: state.items.filter((item) => item.reviewed).length,
        queued: state.items.filter((item) => item.queued_for_briefing).length,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/items") {
      const runId = Number(searchParams.get("run_id") || "0")
      const filtered =
        runId > 0
          ? state.items.filter((item) => item.run_id === runId)
          : state.items
      await jsonResponse(route, {
        items: filtered,
        total: filtered.length,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/outputs") {
      await jsonResponse(route, {
        items: [],
        total: 0,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/templates") {
      await jsonResponse(route, {
        items: [],
        total: 0,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/groups") {
      await jsonResponse(route, {
        items: [],
        total: 0,
        page: pageNum,
        size,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/watchlists/settings") {
      await jsonResponse(route, {
        email_enabled: false,
        default_template_id: null,
        retention_days: 7,
      })
      return
    }

    await route.continue()
  })

  await page.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const { pathname } = url

    if (request.method() === "GET" && pathname === "/api/v1/notifications") {
      await jsonResponse(route, {
        items: state.notifications,
        total: state.notifications.length,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/notifications/unread-count") {
      await jsonResponse(route, {
        unread_count: state.notifications.filter(
          (item) => !item.read_at && !item.dismissed_at
        ).length,
      })
      return
    }

    if (request.method() === "GET" && pathname === "/api/v1/notifications/stream") {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: "",
      })
      return
    }

    await route.continue()
  })
}

test.describe("Watchlist -> Ingest -> Notify journey", () => {
  test("runs a monitor, surfaces the new article, and shows the inbox notification", async ({
    authedPage: page,
    serverInfo,
    diagnostics,
  }) => {
    skipIfServerUnavailable(serverInfo)
    await setupWatchlistJourneyRoutes(page)

    await test.step("Open the monitors section with a seeded monitor", async () => {
      await page.goto("/watchlists", { waitUntil: "domcontentloaded" })
      await waitForConnection(page)

      await expect(page.getByTestId("watchlists-health-bar")).toBeVisible()
      await expect(page.getByTestId("watchlists-repeat-actions")).toBeVisible()
      await expect(page.getByTestId("watchlists-repeat-open-runs")).toBeVisible()

      await page.getByTestId("watchlists-open-command-palette").click()
      await expect(page.getByTestId("watchlists-command-palette-input")).toBeVisible()
      await page.getByTestId("watchlists-command-nav-monitors").click()

      await expect(page.getByLabel(/Monitors table/i)).toBeVisible()
      await expect(page.getByText("Morning Brief")).toBeVisible()
    })

    await test.step("Trigger the real Run Now action", async () => {
      const runRequest = expectApiCall(page, {
        method: "POST",
        url: /\/api\/v1\/watchlists\/jobs\/300\/run$/,
      })

      await page.getByRole("button", { name: /^Run Now$/i }).click()

      const { response } = await runRequest
      expect(response.status()).toBe(200)
    })

    await test.step("Verify the completed run appears in Activity", async () => {
      await page.getByRole("button", { name: "Open Activity" }).click()

      const activitySection = page.getByTestId("watchlists-secondary-activity")
      await expect(page.getByLabel(/Activity runs table/i)).toBeVisible()
      await expect(activitySection.getByText("Morning Brief")).toBeVisible()
      await expect(activitySection.getByRole("button", { name: /Open Reports/i })).toBeVisible()
    })

    await test.step("Verify the ingested article appears in Articles", async () => {
      await page.getByRole("tab", { name: /^(Items|Articles)$/ }).click()

      const row = page.getByTestId("watchlists-item-row-9001")
      await expect(row).toBeVisible()
      await row.click()

      await expect(page.getByTestId("watchlists-item-reader")).toContainText(
        "Morning Brief catches a new article"
      )
      await expect(page.getByTestId("watchlists-item-reader")).toContainText(
        "manual watchlist run fetched one new article"
      )
    })

    await test.step("Verify the notification inbox reflects the completed run", async () => {
      const notificationsPage = new NotificationsPage(page)
      await notificationsPage.goto()
      await notificationsPage.assertPageReady()
      await notificationsPage.waitForLoaded()

      await expect(notificationsPage.notificationsList).toBeVisible()
      await expect(notificationsPage.unreadLabel).toContainText("Unread: 1")
      await expect(page.getByText("Run completed")).toBeVisible()
      await expect(page.getByText("Morning Brief ingested 1 new article.")).toBeVisible()
    })

    await assertNoCriticalErrors(diagnostics)
  })
})
