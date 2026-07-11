import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  getCriticalIssues
} from "../utils/fixtures"
import type { DiagnosticsData } from "../utils/fixtures"
import { stubNotificationsApi, waitForConnection } from "../utils/helpers"

type WatchlistsRouteOptions = {
  watchlists?: Array<Record<string, unknown>>
  sources?: Array<Record<string, unknown>>
  jobs?: Array<Record<string, unknown>>
  runs?: Array<Record<string, unknown>>
  outputs?: Array<Record<string, unknown>>
  latestBriefing?: Record<string, unknown> | null
  briefingByRun?: Record<string, unknown>
  failOutputCreate?: boolean
  showAllViews?: boolean
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

const deterministicSources = [
  {
    ...source,
    id: 101,
    name: "Morning Wire",
    url: "https://example.com/morning.xml",
    tags: ["demo", "news"]
  },
  {
    ...source,
    id: 102,
    name: "City Desk",
    url: "https://example.com/city.xml",
    source_type: "site",
    tags: ["demo", "local"]
  },
  {
    ...source,
    id: 103,
    name: "League Notebook",
    url: "https://example.com/league.xml",
    tags: ["demo", "sports"]
  }
]

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

const readyBriefingProjection = (
  overrides: Record<string, unknown> = {}
): Record<string, unknown> => ({
  occurrence_id: 8801,
  run_id: 404,
  job_id: 303,
  artifact_status: "ready",
  delivery_status: "delivered",
  stages: {
    collect: { status: "ready" },
    select: { status: "ready" },
    render_text: { status: "ready" },
    persist_text: { status: "ready" },
    compose_audio_script: { status: "ready" },
    persist_audio_script: { status: "ready" },
    generate_audio: { status: "ready" },
    persist_audio: { status: "ready" },
    "deliver:email": { status: "ready", outcome: "successful" },
    "deliver:chatbook": { status: "ready", outcome: "successful" }
  },
  output: {
    id: 901,
    title: "Demo Sports Desk",
    created_at: now(),
    metadata: {
      provenance: deterministicSources.map((entry) => ({ source_id: entry.id }))
    }
  },
  audio: {
    run_id: 404,
    task_id: "task_audio_ready",
    status: "completed",
    download_url: "/api/v1/watchlists/runs/404/audio/download",
    script_artifact: {
      artifact_id: "script-404",
      title: "Demo Sports Desk script",
      download_url: "/api/v1/watchlists/runs/404/audio/script/download"
    }
  },
  editorial: {
    program_format: "sportscast",
    outcome_noun: "episode",
    show_name: "Demo Sports Desk",
    show_notes: true,
    target_minutes: 15,
    cast: {
      speaker_count: 2,
      speakers: [
        { label: "Alex", role: "host", voice: "alloy", synthetic: true },
        { label: "Riley", role: "analyst", voice: "nova", synthetic: true }
      ]
    }
  },
  delivery: {
    email: { adapter: "email", recipient_count: 1, masked_label: "demo@example.com" },
    chatbook: { adapter: "chatbook", recipient_count: 1, masked_label: "Demo Chatbook" }
  },
  selection: { candidate_count: 9, included_count: 6, omitted_count: 3, source_count: 3 },
  next_run_at: "2026-07-11T08:00:00-07:00",
  timezone: "America/Los_Angeles",
  recovery: { can_open_report: true, can_regenerate_audio: true },
  ...overrides
})

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
  const showAllViews = options.showAllViews !== false
  await page.addInitScript((enabled) => {
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith("watchlists:")) {
        localStorage.removeItem(key)
      }
    }
    localStorage.setItem("watchlists:show-all-views:v1", enabled ? "true" : "false")
  }, showAllViews)
  const state = {
    watchlists: [...(options.watchlists || [watchlist])],
    sources: [...(options.sources || [])],
    jobs: [...(options.jobs || [])],
    runs: [...(options.runs || [])],
    outputs: [...(options.outputs || [])],
    latestBriefing: options.latestBriefing === undefined ? null : options.latestBriefing,
    briefingByRun: options.briefingByRun || readyBriefingProjection(),
    sourceTests: [] as Array<Record<string, unknown>>,
    createdSources: [] as Array<Record<string, unknown>>,
    createdJobs: [] as Array<Record<string, unknown>>,
    updatedJobs: [] as Array<Record<string, unknown>>,
    runTriggers: [] as Array<number>,
    briefingRetries: [] as Array<Record<string, unknown>>,
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

    if (method === "POST" && pathname === "/api/v1/watchlists") {
      const payload = request.postDataJSON() as Record<string, unknown>
      const created = {
        ...watchlist,
        ...payload,
        id: 43,
        created_at: now(),
        updated_at: now()
      }
      state.watchlists = [created, ...state.watchlists]
      await jsonResponse(route, created)
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
        id: 501 + state.createdSources.length - 1,
        status: "ok",
        created_at: now(),
        updated_at: now()
      }
      state.sources = [...state.sources, created]
      await jsonResponse(route, created)
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/sources/test") {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.sourceTests.push(payload)
      await jsonResponse(route, {
        items: [
          {
            source_id: Number(payload.source_id || 501),
            source_type: payload.source_type || "rss",
            url: `${payload.url || "https://example.com"}/story`,
            title: `${payload.name || "Demo"} candidate`,
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
        id: 303 + state.createdJobs.length - 1,
        created_at: now(),
        updated_at: now()
      }
      state.jobs = [created]
      await jsonResponse(route, created)
      return
    }

    const jobUpdateMatch = pathname.match(/^\/api\/v1\/watchlists\/jobs\/(\d+)$/)
    if (method === "PATCH" && jobUpdateMatch) {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.updatedJobs.push({ id: Number(jobUpdateMatch[1]), ...payload })
      state.jobs = state.jobs.map((entry) =>
        Number(entry.id) === Number(jobUpdateMatch[1])
          ? { ...entry, ...payload, updated_at: now() }
          : entry
      )
      await jsonResponse(route, state.jobs[0] || { ...job, id: Number(jobUpdateMatch[1]), ...payload })
      return
    }

    const jobRunMatch = pathname.match(/^\/api\/v1\/watchlists\/jobs\/(\d+)\/run$/)
    if (method === "POST" && jobRunMatch) {
      state.runTriggers.push(Number(jobRunMatch[1]))
      state.runs = [completedRun]
      state.latestBriefing = state.briefingByRun
      await jsonResponse(route, completedRun)
      return
    }

    if (method === "POST" && pathname === "/api/v1/watchlists/schedule/preview") {
      await jsonResponse(route, {
        next_run_at: "2026-07-12T18:00:00-07:00",
        following_run_at: "2026-07-19T18:00:00-07:00"
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/briefings/latest") {
      if (state.latestBriefing) {
        await jsonResponse(route, state.latestBriefing)
      } else {
        await jsonResponse(route, { detail: "not found" }, 404)
      }
      return
    }

    const runBriefingMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/briefing$/)
    if (method === "GET" && runBriefingMatch) {
      await jsonResponse(route, state.briefingByRun)
      return
    }

    if (method === "POST" && runBriefingMatch) {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.briefingRetries.push(payload)
      state.briefingByRun = readyBriefingProjection()
      state.latestBriefing = state.briefingByRun
      await jsonResponse(route, state.briefingByRun)
      return
    }

    const runBriefingRetryMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/briefing\/retry$/)
    if (method === "POST" && runBriefingRetryMatch) {
      const payload = request.postDataJSON() as Record<string, unknown>
      state.briefingRetries.push(payload)
      state.briefingByRun = readyBriefingProjection()
      state.latestBriefing = state.briefingByRun
      await jsonResponse(route, state.briefingByRun)
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/runs") {
      const q = searchParams.get("q")
      const filtered = q
        ? state.runs.filter((run) => String(run.status || "") === q)
        : state.runs
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

    if (method === "GET" && pathname === "/api/v1/watchlists/runs/404/audio/download") {
      await route.fulfill({
        status: 200,
        contentType: "audio/mpeg",
        body: "fixture-audio"
      })
      return
    }

    if (method === "GET" && pathname === "/api/v1/watchlists/runs/404/audio/script/download") {
      await route.fulfill({
        status: 200,
        contentType: "text/markdown",
        body: "# Demo Sports Desk\n\nShow notes text"
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
      const row = page.getByRole("row").filter({ hasText: expected.title }).first()
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

  test("proves canonical briefing flow, latest episode, recovery, and sportscast preset", async ({
    authedPage: page,
    diagnostics
  }) => {
    const failedAudioProjection = readyBriefingProjection({
      artifact_status: "failed",
      delivery_status: "waiting_for_artifacts",
      audio: { run_id: 404, status: "failed", error: "TTS provider timeout" },
      stages: {
        ...(readyBriefingProjection().stages as Record<string, unknown>),
        generate_audio: { status: "failed", retryable: true, code: "provider_timeout" },
        persist_audio: { status: "not_started" }
      },
      recovery: { can_open_report: true, can_retry_audio: true }
    })
    const state = await setupWatchlistsReadinessRoutes(page, {
      sources: deterministicSources,
      latestBriefing: null,
      briefingByRun: failedAudioProjection,
      showAllViews: false
    })

    await page.goto("/watchlists?tab=overview", { waitUntil: "domcontentloaded" })
    await waitForConnection(page)

    await page.getByRole("button", { name: "Set up briefing" }).click()
    const wizard = page.getByRole("dialog", { name: "Set up briefing" })
    await expect(wizard).toBeVisible()

    await expect(wizard.getByRole("navigation", { name: "Briefing setup steps" })).toContainText("Sources")
    for (const fixtureSource of deterministicSources) {
      await wizard.getByText(fixtureSource.name, { exact: true }).click()
    }
    await wizard.getByRole("button", { name: "Test source" }).click()
    await expect(wizard.getByText(/Ready/).first()).toBeVisible()
    await page.getByTestId("watchlists-pipeline-next-step").click()

    await wizard.getByLabel("Monitor name").fill("Demo Sportscast")
    await expect(wizard.getByText(/Saturday, July 11 at 8:00 AM/)).toBeVisible()
    await expect(wizard.getByText("America/Los_Angeles")).toBeVisible()
    await page.getByTestId("watchlists-pipeline-next-step").click()

    await wizard.getByRole("radio", { name: "Sportscast" }).check()
    await wizard.getByLabel("Show name").fill("Demo Sports Desk")
    await wizard.getByLabel("Target duration in minutes").fill("15")
    await wizard.getByLabel("Target duration in minutes").blur()
    await wizard.getByLabel("Cast size").click()
    await page
      .locator(".ant-select-dropdown:not(.ant-select-dropdown-hidden) .ant-select-item-option-content")
      .getByText("2", { exact: true })
      .click()
    await expect(wizard.getByLabel("Speaker 2 label")).toBeVisible()
    await wizard.getByLabel("Speaker 1 label").fill("Alex")
    await wizard.getByLabel("Speaker 2 label").fill("Riley")
    await page.getByTestId("watchlists-pipeline-next-step").click()

    const emailSwitch = wizard.getByRole("switch", { name: "Email" })
    await emailSwitch.click()
    await expect(emailSwitch).toBeChecked()
    await wizard.getByLabel("Email recipients").fill("demo@example.com")
    await wizard.getByLabel("Email recipients").blur()
    const chatbookSwitch = wizard.getByRole("switch", { name: "Chatbook" })
    await chatbookSwitch.click()
    await expect(chatbookSwitch).toBeChecked()
    await wizard.getByLabel("Chatbook title").fill("Demo Chatbook")
    await wizard.getByLabel("Chatbook title").blur()
    await page.getByTestId("watchlists-pipeline-next-step").click()

    const receipt = wizard.getByTestId("watchlists-pipeline-receipt")
    await expect(receipt).toContainText(/Saturday, July 11 at 8:00 AM/)
    await expect(receipt).toContainText("America/Los_Angeles")
    await expect(receipt).toContainText("3 sources")
    await expect(receipt).toContainText("text")
    await expect(receipt).toContainText("targeting 15 minutes")
    await expect(receipt).toContainText("Reports")
    await expect(receipt).toContainText("Email")
    await expect(receipt).toContainText("Chatbook")

    await wizard.getByRole("button", { name: "Generate 60-second sample" }).click()
    await expect(wizard.getByText(/Create audio failed/)).toBeVisible()
    await expect.poll(() => state.createdJobs.length).toBe(1)
    expect(state.createdJobs[0]).toMatchObject({
      name: "Demo Sportscast",
      active: false,
      scope: { sources: [101, 102, 103] }
    })
    expect(state.createdJobs[0].output_prefs).toMatchObject({
      briefing_pipeline: {
        version: 1,
        editorial: {
          program_format: "sportscast",
          outcome_noun: "episode",
          show_name: "Demo Sports Desk"
        },
        text: { enabled: true, template_name: "briefing_markdown" },
        audio: {
          enabled: true,
          target_minutes: 1,
          cast: {
            speaker_count: 2
          }
        },
        delivery: {
          reports: { enabled: true },
          email: { enabled: false, recipients: [] },
          chatbook: { enabled: false, title: "Demo Chatbook" }
        },
        test: { external_delivery: false, audio_sample_seconds: 60 }
      }
    })
    expect(state.createdJobs[0].output_prefs).not.toHaveProperty("generate_audio")
    expect(state.createdJobs[0].output_prefs).not.toHaveProperty("target_audio_minutes")
    expect(state.runTriggers).toEqual([303])

    await wizard.getByRole("button", { name: "Activate schedule" }).click()
    await expect.poll(() => state.updatedJobs.some((entry) => entry.active === true)).toBe(true)
    const scheduledJobUpdate = state.updatedJobs.find((entry) => entry.output_prefs)
    expect(scheduledJobUpdate?.output_prefs).toMatchObject({
      briefing_pipeline: {
        audio: { enabled: true, target_minutes: 15, cast: { speaker_count: 2 } },
        delivery: {
          reports: { enabled: true },
          email: { enabled: true, recipients: ["demo@example.com"] },
          chatbook: { enabled: true, title: "Demo Chatbook" }
        }
      }
    })
    await expect(wizard).toBeHidden()

    const latestBriefing = page.getByRole("region", { name: "Latest episode" })
    await expect(latestBriefing.getByRole("heading", { name: "Latest episode" })).toBeVisible()
    await expect(latestBriefing.getByText("Demo Sports Desk")).toBeVisible()
    await expect(latestBriefing.getByText("audio failed")).toBeVisible()
    await page.getByRole("button", { name: "Retry generating audio for Demo Sports Desk" }).click()
    await expect.poll(() => state.briefingRetries).toHaveLength(1)
    expect(state.briefingRetries[0]).toMatchObject({ stage: "generate_audio" })
    expect(state.createdSources).toHaveLength(0)

    await expect(latestBriefing.getByRole("button", { name: "Play Demo Sports Desk" })).toBeVisible()
    await expect(latestBriefing.getByRole("button", { name: "View all reports" })).toBeVisible()
    await expect(latestBriefing.getByText("Email delivered")).toBeVisible()
    await expect(latestBriefing.getByText("Chatbook delivered")).toBeVisible()
    await expect(latestBriefing.getByText("3 tracked sources")).toBeVisible()
    await expect(latestBriefing.getByText("Included 6")).toBeVisible()
    await expect(latestBriefing.getByText(/Next run: Saturday, July 11 at 8:00 AM/)).toBeVisible()
    await page.getByRole("button", { name: "Review script for Demo Sports Desk" }).click()
    const scriptDialog = page.getByRole("dialog", { name: "Demo Sports Desk script" })
    await expect(scriptDialog).toContainText("Show notes text")
    await scriptDialog.getByRole("button", { name: "Close" }).click()
    await expect(page.getByRole("button", { name: "Open show notes: Demo Sports Desk" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Inspect run 404: Demo Sports Desk" })).toBeVisible()
    await page.getByRole("button", { name: "Test now: Demo Sports Desk" }).click()
    await expect.poll(() => state.runTriggers.length).toBeGreaterThanOrEqual(2)

    assertNoUnmatchedWatchlistsRequests(state)
    await assertNoRuntimeOverlay(page)
    await assertNoUnexpectedCriticalErrors(diagnostics, {
      allowedConsoleErrorPatterns: [
        /Failed to load resource: the server responded with a status of 404 \(Not Found\)/
      ]
    })
  })
})
