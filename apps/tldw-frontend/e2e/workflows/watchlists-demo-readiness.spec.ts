import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  getCriticalIssues
} from "../utils/fixtures"
import type { DiagnosticsData } from "../utils/fixtures"
import { stubNotificationsApi, waitForConnection } from "../utils/helpers"

type WatchlistsRouteOptions = {
  sources?: Array<Record<string, unknown>>
  jobs?: Array<Record<string, unknown>>
  runs?: Array<Record<string, unknown>>
  outputs?: Array<Record<string, unknown>>
  failOutputCreate?: boolean
}

const now = () => "2026-05-20T15:00:00Z"

const watchlist = {
  id: 42,
  name: "Demo News Watchlist",
  description: "Demo collection for briefing readiness checks",
  objective: "Track demo news sources and generate briefings",
  domain: "news",
  status: "active",
  priority: "medium",
  tags: ["demo"],
  archived_at: null,
  deleted_at: null,
  restore_expires_at: null,
  created_at: now(),
  updated_at: now()
}

const source = {
  id: 101,
  name: "Demo Feed",
  url: "https://example.com/feed.xml",
  source_type: "rss",
  active: true,
  status: "ok",
  tags: ["demo"],
  created_at: now(),
  updated_at: now()
}

const job = {
  id: 303,
  name: "Demo Briefing",
  description: "Demo monitor",
  active: true,
  scope: { sources: [101], groups: [], tags: [] },
  schedule_expr: "0 8 * * *",
  timezone: "UTC",
  job_filters: { filters: [] },
  output_prefs: {
    auto_output: { enabled: true },
    template_name: "briefing_markdown",
    template: { default_name: "briefing_markdown" },
    generate_audio: true
  },
  created_at: now(),
  updated_at: now(),
  last_run_at: null,
  next_run_at: now()
}

const completedRun = {
  id: 404,
  job_id: 303,
  status: "completed",
  started_at: now(),
  finished_at: now(),
  stats: {
    items_found: 2,
    items_ingested: 2,
    items_filtered: 0,
    items_errored: 0
  }
}

const audioOutputs = [
  {
    id: 701,
    run_id: 404,
    job_id: 303,
    type: "briefing",
    format: "md",
    title: "Pending audio report",
    version: 1,
    expired: false,
    metadata: {
      template_name: "briefing_markdown",
      audio_briefing_requested: true,
      audio_briefing_status: "pending",
      audio_briefing_task_id: "task_audio_pending"
    },
    created_at: now(),
    expires_at: null
  },
  {
    id: 702,
    run_id: 404,
    job_id: 303,
    type: "briefing",
    format: "md",
    title: "Failed audio report",
    version: 1,
    expired: false,
    metadata: {
      template_name: "briefing_markdown",
      audio_briefing_requested: true,
      audio_briefing_status: "failed",
      audio_briefing_error: "TTS provider timeout"
    },
    created_at: now(),
    expires_at: null
  },
  {
    id: 703,
    run_id: 404,
    job_id: 303,
    type: "briefing",
    format: "md",
    title: "Skipped audio report",
    version: 1,
    expired: false,
    metadata: {
      template_name: "briefing_markdown",
      audio_briefing_requested: true,
      audio_briefing_status: "skipped",
      audio_briefing_error: "no briefing text"
    },
    created_at: now(),
    expires_at: null
  }
]

const jsonResponse = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload)
  })
}

const assertNoRuntimeOverlay = async (page: Page) => {
  await expect(page.getByText(/Unhandled Runtime Error|Build Error|Application error/i)).toHaveCount(0)
}

const assertNoUnexpectedCriticalErrors = async (
  diagnostics: DiagnosticsData,
  options: { allowedConsoleErrorPatterns?: RegExp[] } = {}
) => {
  const critical = getCriticalIssues(diagnostics)
  const allowedConsoleErrorPatterns = [
    /Warning: \[antd: message\] Static function can not consume context like dynamic theme/,
    /WebSocket connection to 'ws:\/\/127\.0\.0\.1:8000\/api\/v1\/watchlists\/runs\/404\/stream\?api_key=[^']+' failed: Error in connection establishment: net::ERR_CONNECTION_REFUSED/,
    ...(options.allowedConsoleErrorPatterns || [])
  ]
  const consoleErrors = critical.consoleErrors.filter(
    (entry) => !allowedConsoleErrorPatterns.some((pattern) => pattern.test(entry.text))
  )

  if (
    critical.pageErrors.length > 0 ||
    consoleErrors.length > 0 ||
    critical.requestFailures.length > 0
  ) {
    throw new Error(
      `Unexpected browser diagnostics:\n${JSON.stringify(
        {
          pageErrors: critical.pageErrors,
          consoleErrors,
          requestFailures: critical.requestFailures
        },
        null,
        2
      )}`
    )
  }
}

const assertNoUnmatchedWatchlistsRequests = (state: {
  unmatchedRequests: Array<{ method: string; path: string }>
}) => {
  expect(state.unmatchedRequests).toEqual([])
}

const EXPECTED_REGENERATE_FAILURE_CONSOLE = /Failed to regenerate output:.*template_not_found: briefing_markdown/
const EXPECTED_OUTPUT_CREATE_400_CONSOLE =
  /Failed to load resource: the server responded with a status of 400 \(Bad Request\)/

const fillGuidedQuickSetup = async (page: Page) => {
  const quickSetupDialog = page.getByRole("dialog", { name: "Add initial collection" })
  const openedAutomatically = await quickSetupDialog
    .waitFor({ state: "visible", timeout: 2000 })
    .then(() => true)
    .catch(() => false)
  if (!openedAutomatically) {
    await page.getByTestId("watchlists-overview-cta-guided-setup").click()
  }
  await expect(quickSetupDialog).toBeVisible()
  await quickSetupDialog.getByLabel("Feed name").fill("Demo Feed")
  await quickSetupDialog.getByRole("textbox", { name: "* Feed URL" }).fill("https://example.com/feed.xml")
  await quickSetupDialog.getByRole("button", { name: "Next" }).click()

  await quickSetupDialog.getByLabel("Monitor name").fill("Demo Briefing")
  await quickSetupDialog.getByRole("button", { name: "Next" }).click()

  await expect(quickSetupDialog.getByTestId("watchlists-overview-quick-setup-candidate-summary"))
    .toContainText("1 ingestable")
  await quickSetupDialog.getByRole("button", { name: /Create collection/i }).click()
}

const setupWatchlistsReadinessRoutes = async (
  page: Page,
  options: WatchlistsRouteOptions = {}
) => {
  await page.addInitScript(() => {
    localStorage.setItem("watchlists:show-all-views:v1", "true")
  })
  const state = {
    watchlists: [watchlist],
    sources: [...(options.sources || [])],
    jobs: [...(options.jobs || [])],
    runs: [...(options.runs || [])],
    outputs: [...(options.outputs || [])],
    sourceTests: [] as Array<Record<string, unknown>>,
    createdSources: [] as Array<Record<string, unknown>>,
    createdJobs: [] as Array<Record<string, unknown>>,
    outputCreates: [] as Array<Record<string, unknown>>,
    unmatchedRequests: [] as Array<{ method: string; path: string }>
  }

  await page.route(/\/api\/v1\/persona\/profiles(?:\?.*)?$/, async (route) => {
    await jsonResponse(route, [])
  })

  await page.route(/\/api\/v1\/watchlists(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const { pathname, searchParams } = url
    const method = request.method()
    const pageNum = Number(searchParams.get("page") || "1")
    const size = Number(searchParams.get("size") || "25")

    if (method === "GET" && pathname === "/api/v1/watchlists") {
      await jsonResponse(route, {
        items: state.watchlists,
        total: state.watchlists.length,
        page: pageNum,
        size
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/sources") {
      await jsonResponse(route, {
        items: state.sources,
        total: state.sources.length,
        page: pageNum,
        size
      })
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/sources") {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.createdSources.push(payload)
      const created = {
        ...source,
        ...payload,
        id: 501,
        status: "ok",
        created_at: now(),
        updated_at: now()
      }
      state.sources = [created]
      await jsonResponse(route, created)
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/sources/test") {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.sourceTests.push(payload)
      await jsonResponse(route, {
        items: [
          {
            source_id: 501,
            source_type: "rss",
            url: "https://example.com/story",
            title: "Demo source candidate",
            summary: "Candidate item",
            decision: "ingest"
          }
        ],
        total: 1,
        ingestable: 1,
        filtered: 0
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/jobs") {
      await jsonResponse(route, {
        items: state.jobs,
        total: state.jobs.length,
        page: pageNum,
        size
      })
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/jobs") {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.createdJobs.push(payload)
      const created = {
        ...job,
        ...payload,
        id: 303,
        created_at: now(),
        updated_at: now()
      }
      state.jobs = [created]
      await jsonResponse(route, created)
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/jobs/303/run") {
      state.runs = [completedRun]
      await jsonResponse(route, completedRun)
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/runs") {
      const q = searchParams.get("q")
      const filtered = q ? state.runs.filter((run) => run.status === q) : state.runs
      await jsonResponse(route, {
        items: filtered,
        total: filtered.length,
        page: pageNum,
        size
      })
      return
    }

    const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
    if (method === "GET" && runDetailsMatch) {
      await jsonResponse(route, {
        ...completedRun,
        filter_tallies: { include: 2 },
        log_text: "Completed successfully",
        log_path: null,
        truncated: false,
        filtered_sample: null
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/runs/404/audio") {
      await jsonResponse(route, {
        run_id: 404,
        status: "pending",
        task_id: "task_audio_pending"
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/items") {
      await jsonResponse(route, {
        items: [],
        total: 0,
        page: pageNum,
        size
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/items/smart-counts") {
      await jsonResponse(route, {
        all: 0,
        today: 0,
        today_unread: 0,
        unread: 0,
        reviewed: 0,
        queued: 0
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/outputs") {
      await jsonResponse(route, {
        items: state.outputs,
        total: state.outputs.length,
        page: pageNum,
        size
      })
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/outputs") {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.outputCreates.push(payload)
      if (options.failOutputCreate) {
        await jsonResponse(
          route,
          {
            detail: {
              code: "template_not_found",
              message: "template_not_found: briefing_markdown"
            }
          },
          400
        )
        return
      }
      const created = {
        id: 901,
        run_id: payload.run_id,
        job_id: 303,
        type: "briefing",
        format: "md",
        title: "Created report",
        version: 1,
        expired: false,
        metadata: payload,
        created_at: now(),
        expires_at: null
      }
      state.outputs = [created, ...state.outputs]
      await jsonResponse(route, created)
      return
    }

    const outputDownloadMatch = pathname.match(/^\/api\/v1\/watchlists\/outputs\/(\d+)\/download$/)
    if (method === "GET" && outputDownloadMatch) {
      await route.fulfill({
        status: 200,
        contentType: "text/markdown",
        body: "# Demo briefing\n\nBody text"
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/templates") {
      await jsonResponse(route, {
        items: [{ name: "briefing_markdown", format: "md", updated_at: now() }],
        total: 1,
        page: pageNum,
        size
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/settings") {
      await jsonResponse(route, {
        default_output_ttl_seconds: 86400,
        temporary_output_ttl_seconds: 3600
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/groups") {
      await jsonResponse(route, { items: [], total: 0, page: pageNum, size })
      return
    }

    const alertsMatch = pathname.match(/^\/api\/v1\/watchlists\/(\d+)\/alerts$/)
    if (method === "GET" && alertsMatch) {
      await jsonResponse(route, { items: [], total: 0, page: pageNum, size })
      return
    }

    if (method === "POST" && pathname.startsWith("/api/v1/watchlists/telemetry/")) {
      await jsonResponse(route, { ok: true })
      return
    }

    state.unmatchedRequests.push({ method, path: `${pathname}${url.search}` })
    await jsonResponse(
      route,
      {
        detail: {
          code: "unmatched_watchlists_mock",
          message: `Unhandled Watchlists mock route: ${method} ${pathname}`
        }
      },
      500
    )
  })
  await stubNotificationsApi(page)

  return state
}

test.describe("Watchlists demo readiness gate", () => {
  test("preflights and creates the first guided news feed and monitor", async ({
    authedPage: page,
    diagnostics
  }) => {
    const state = await setupWatchlistsReadinessRoutes(page)

    await page.goto("/watchlists?tab=overview", { waitUntil: "domcontentloaded" })
    await waitForConnection(page)

    await expect(page.getByRole("heading", { name: "Watchlists" })).toBeVisible()
    await assertNoRuntimeOverlay(page)

    await fillGuidedQuickSetup(page)

    await expect.poll(() => state.sourceTests.length).toBe(1)
    expect(state.sourceTests[0]).toMatchObject({
      url: "https://example.com/feed.xml",
      source_type: "rss"
    })
    await expect.poll(() => state.createdSources.length).toBe(1)
    expect(state.createdSources[0]).toMatchObject({
      name: "Demo Feed",
      url: "https://example.com/feed.xml",
      source_type: "rss",
      active: true
    })
    await expect.poll(() => state.createdJobs.length).toBe(1)
    expect(state.createdJobs[0]).toMatchObject({
      name: "Demo Briefing",
      scope: { sources: [501] },
      output_prefs: {
        template_name: "briefing_markdown",
        template: { default_name: "briefing_markdown" },
        generate_audio: true
      }
    })
    await expect.poll(() => state.runs.length).toBe(1)

    assertNoUnmatchedWatchlistsRequests(state)
    await assertNoRuntimeOverlay(page)
    await assertNoUnexpectedCriticalErrors(diagnostics)
  })

  test("loads /watchlists and creates the demo briefing monitor with backend template names", async ({
    authedPage: page,
    diagnostics
  }) => {
    const state = await setupWatchlistsReadinessRoutes(page, {
      sources: [source]
    })

    await page.goto("/watchlists?tab=overview", { waitUntil: "domcontentloaded" })
    await waitForConnection(page)

    await expect(page.getByRole("heading", { name: "Watchlists" })).toBeVisible()
    await assertNoRuntimeOverlay(page)

    await page.getByTestId("watchlists-overview-cta-pipeline-builder").click()
    const pipelineDialog = page.getByRole("dialog", { name: "Briefing pipeline builder" })
    await expect(pipelineDialog).toBeVisible()
    const demoFeedCheckbox = pipelineDialog.getByRole("checkbox", { name: "Demo Feed" })
    await expect(demoFeedCheckbox).toBeVisible()
    await demoFeedCheckbox.check()
    await expect(demoFeedCheckbox).toBeChecked()
    await pipelineDialog.getByRole("button", { name: "Next" }).click()

    await pipelineDialog.getByLabel("Monitor name").fill("Demo Briefing")
    await expect(pipelineDialog.getByLabel("Monitor name")).toHaveValue("Demo Briefing")
    await pipelineDialog.getByRole("button", { name: "Next" }).click()

    await expect(pipelineDialog.getByLabel("Template")).toHaveValue("briefing_md")
    await pipelineDialog.getByRole("button", { name: "Next" }).click()

    await expect(pipelineDialog.getByLabel("Audio briefing")).toBeVisible()
    await pipelineDialog.getByRole("button", { name: "Next" }).click()

    await expect(pipelineDialog.getByTestId("watchlists-pipeline-review-summary")).toBeVisible()
    await pipelineDialog.getByRole("button", { name: "Create pipeline" }).click()

    await expect.poll(() => state.createdJobs.length).toBe(1)
    expect(state.createdJobs[0]).toMatchObject({
      name: "Demo Briefing",
      scope: { sources: [101] },
      output_prefs: {
        template_name: "briefing_markdown",
        template: { default_name: "briefing_markdown" },
        generate_audio: true
      }
    })
    await expect.poll(() => state.outputCreates.length).toBe(1)
    expect(state.outputCreates[0]).toMatchObject({
      run_id: 404,
      template_name: "briefing_markdown",
      generate_audio: true
    })

    assertNoUnmatchedWatchlistsRequests(state)
    await assertNoRuntimeOverlay(page)
    await assertNoUnexpectedCriticalErrors(diagnostics)
  })

  test("renders output creation failure in-app and keeps audio status truthful", async ({
    authedPage: page,
    diagnostics
  }) => {
    const state = await setupWatchlistsReadinessRoutes(page, {
      sources: [source],
      jobs: [job],
      runs: [completedRun],
      outputs: audioOutputs,
      failOutputCreate: true
    })

    await page.goto("/watchlists?tab=outputs", { waitUntil: "domcontentloaded" })
    await waitForConnection(page)

    await page.getByRole("tab", { name: /Reports/ }).click()
    await expect(page.getByText("Pending audio report")).toBeVisible()
    await page.getByRole("button", { name: "Regenerate" }).first().click()
    const regenerateDialog = page.getByRole("dialog", { name: "Regenerate Output" })
    await regenerateDialog.getByRole("button", { name: "Regenerate" }).click()

    await expect(page.getByTestId("watchlists-outputs-live-region")).toContainText(
      "Failed to regenerate Pending audio report."
    )
    await regenerateDialog.getByRole("button", { name: "Cancel" }).click()
    await expect.poll(() => state.outputCreates.length).toBe(1)
    expect(state.outputCreates[0]).toMatchObject({
      run_id: 404,
      template_name: "briefing_markdown"
    })
    await assertNoRuntimeOverlay(page)

    for (const expected of [
      { title: "Pending audio report", status: "Queued", detail: "task_audio_pending" },
      { title: "Failed audio report", status: "Failed", detail: "TTS provider timeout" },
      { title: "Skipped audio report", status: "Skipped", detail: "no briefing text" }
    ]) {
      const row = page.locator(".ant-table-row").filter({ hasText: expected.title })
      await expect(row).toBeVisible()
      await row.getByRole("button", { name: "Preview" }).click()
      const previewDrawer = page.getByRole("dialog", { name: expected.title })
      await expect(previewDrawer).toContainText(expected.status)
      await expect(previewDrawer).toContainText(expected.detail)
      await previewDrawer.getByRole("button", { name: "Close" }).click()
    }

    assertNoUnmatchedWatchlistsRequests(state)
    await assertNoRuntimeOverlay(page)
    await assertNoUnexpectedCriticalErrors(diagnostics, {
      allowedConsoleErrorPatterns: [
        EXPECTED_REGENERATE_FAILURE_CONSOLE,
        EXPECTED_OUTPUT_CREATE_400_CONSOLE
      ]
    })
  })
})
