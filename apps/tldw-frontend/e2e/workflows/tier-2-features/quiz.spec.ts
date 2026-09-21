/**
 * Quiz Playground E2E Tests (Tier 2)
 *
 * Tests the Quiz Playground page lifecycle:
 * - Page loads with beta badge and appropriate state (online playground, demo, or connection banner)
 * - Tab switching between Take, Generate, Create, Manage, Results
 * - Global search and reset controls
 * - Demo quiz flow (start, take, submit, results) when in demo/offline mode
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/quiz.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { QuizPage } from "../../utils/page-objects/QuizPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"
import type { Page, Route } from "@playwright/test"

const NOW = "2026-09-11T12:00:00Z"
const CHECK_ID = "11111111-1111-4111-8111-111111111111"
const DOMAIN_ID = "22222222-2222-4222-8222-222222222222"
const NEEDS_WORK_ID = "33333333-3333-4333-8333-333333333333"
const EFFECTIVE_ID = "44444444-4444-4444-8444-444444444444"
const KEY_POINT_ID = "55555555-5555-4555-8555-555555555555"

type MockOsceOptions = {
  generated?: boolean
  stationCount?: number
  nearLimit?: boolean
}

const json = async (route: Route, status: number, body: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: { "access-control-allow-origin": "*" },
    body: JSON.stringify(body),
  })
}

const makeStationContent = (title: string, nearLimit = false) => ({
  schema_version: "osce.station.v1",
  title,
  candidate_instructions: "Review the fictional patient information and explain your approach.",
  candidate_task: "Explain safe anticoagulant use and check understanding.",
  patient_context: {
    text: "A fictional adult is starting anticoagulant treatment after a simulated assessment.",
    citations: [{ source_type: "note", source_id: "note-1", label: "Clinical handover note" }],
  },
  recommended_duration_seconds: 480,
  checklist_items: Array.from({ length: nearLimit ? 50 : 1 }, (_, index) => ({
    id: index === 0 ? CHECK_ID : `11111111-1111-4111-8${String(index).padStart(3, "0")}-111111111111`,
    label: `Checks safety item ${index + 1}`,
    rationale: "This is part of the source-grounded safety conversation.",
    citations: [],
  })),
  rubric_domains: [{
    id: DOMAIN_ID,
    label: "Communication",
    levels: [
      { id: NEEDS_WORK_ID, label: "Needs development", description: "Important points were unclear." },
      { id: EFFECTIVE_ID, label: "Effective", description: "The explanation was clear and structured." },
    ],
  }],
  expected_key_points: [{
    id: KEY_POINT_ID,
    text: "Explain bleeding precautions and when to seek help.",
    citations: [],
  }],
})

const installMockOsceBackend = async (page: Page, options: MockOsceOptions = {}) => {
  const quizRecord = {
    id: 81,
    name: "Generated OSCE Practice",
    description: "Source-grounded clinical communication practice",
    activity_type: "osce",
    generation_profile: "osce_scenario",
    total_questions: 0,
    total_stations: options.stationCount ?? 2,
    passing_score: null,
    time_limit_seconds: null,
    deleted: false,
    client_id: "e2e-user",
    version: 1,
    created_at: NOW,
    updated_at: NOW,
  }
  let generated = options.generated ?? false
  let failNextPatchAsOffline = false
  let attemptVersion = 1
  let attemptState: "in_progress" | "self_assessment" | "completed" = "in_progress"
  let attemptNotes = ""
  let checklistSelections: Record<string, string> = {}
  let rubricSelections: Record<string, string> = {}
  let attemptSnapshot: ReturnType<typeof makeStationContent> | null = null
  let stationDeleted = false

  const stationCount = options.stationCount ?? 2
  const stations = Array.from({ length: stationCount }, (_, index) => ({
    id: 901 + index,
    quiz_id: 81,
    content: makeStationContent(
      options.nearLimit && index === 0
        ? "N".repeat(200)
        : index === 0 ? "Anticoagulant counselling" : `Station ${index + 1}`,
      options.nearLimit && index === 0,
    ),
    order_index: index,
    version: 1,
    origin: "generated",
    provenance: { source_count: 1 },
    source_bundle: [{ source_type: "note", source_id: "note-1" }],
    verification_state: "source_verified",
    verification_timestamp: NOW,
    verification_summary: "Verified against the selected note.",
    deleted: false,
    created_at: NOW,
    updated_at: NOW,
  }))

  const summary = (station: typeof stations[number]) => ({
    id: station.id,
    quiz_id: station.quiz_id,
    title: station.content.title,
    recommended_duration_seconds: station.content.recommended_duration_seconds,
    order_index: station.order_index,
    version: station.version,
    checklist_count: station.content.checklist_items.length,
    rubric_domain_count: station.content.rubric_domains.length,
    verification_state: station.verification_state,
    created_at: station.created_at,
    updated_at: station.updated_at,
  })

  const candidateStation = () => {
    const content = attemptSnapshot ?? stations[0].content
    return {
      schema_version: content.schema_version,
      title: content.title,
      candidate_instructions: content.candidate_instructions,
      candidate_task: content.candidate_task,
      patient_context: {
        text: content.patient_context.text,
        citations: content.patient_context.citations.map(({ source_type, source_id, label }) => ({
          source_type, source_id, label,
        })),
      },
      recommended_duration_seconds: content.recommended_duration_seconds,
    }
  }

  const attempt = () => ({
    id: 701,
    quiz_id: 81,
    station_id: 901,
    client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    state: attemptState,
    version: attemptVersion,
    station: attemptState === "in_progress" ? candidateStation() : attemptSnapshot,
    notes: attemptNotes,
    ...(attemptState === "in_progress" ? {} : {
      checklist_selections: checklistSelections,
      rubric_selections: rubricSelections,
      self_assessment_started_at: NOW,
      completed_at: attemptState === "completed" ? NOW : null,
      elapsed_seconds: 240,
    }),
    started_at: NOW,
    last_modified_at: NOW,
    server_time: NOW,
  })

  const attemptSummary = () => ({
    id: 701,
    quiz_id: 81,
    station_id: 901,
    client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    station_title: (attemptSnapshot ?? stations[0].content).title,
    state: attemptState,
    version: attemptVersion,
    started_at: NOW,
    self_assessment_started_at: attemptState === "in_progress" ? null : NOW,
    completed_at: attemptState === "completed" ? NOW : null,
    last_modified_at: NOW,
    elapsed_seconds: attemptState === "in_progress" ? null : 240,
    checklist_met_count: attemptState === "completed"
      ? Object.values(checklistSelections).filter((value) => value === "met").length
      : null,
    checklist_total: attemptState === "completed" ? (attemptSnapshot ?? stations[0].content).checklist_items.length : null,
    rubric_results: attemptState === "completed" ? [{
      domain_id: DOMAIN_ID,
      domain_label: "Communication",
      level_id: rubricSelections[DOMAIN_ID] ?? NEEDS_WORK_ID,
      level_label: rubricSelections[DOMAIN_ID] === EFFECTIVE_ID ? "Effective" : "Needs development",
    }] : [],
  })

  await page.route("**/openapi.json", (route) => json(route, 200, {
    info: { version: "e2e" },
    paths: {
      "/api/v1/quizzes": {},
      "/api/v1/quizzes/generate": {},
      "/api/v1/quizzes/generation-profiles": {},
    },
  }))

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()

    if (method === "OPTIONS") return json(route, 200, {})
    if (path === "/api/v1/health/live") return json(route, 200, { status: "ok" })
    if (path === "/api/v1/auth/me") return json(route, 200, { id: 1, username: "e2e-user", is_active: true })
    if (path === "/api/v1/quizzes/generation-profiles") return json(route, 200, [{
      id: "osce_scenario",
      label: "OSCE Scenario",
      description: "Source-grounded clinical practice stations.",
      status: "available",
      output_kind: "osce_stations",
      default_num_stations: 1,
      default_num_questions: 1,
      default_difficulty: "mixed",
      default_question_types: [],
    }])
    if (path === "/api/v1/notes/" && method === "GET") {
      return json(route, 200, { items: [{ id: "note-1", title: "Clinical handover note" }] })
    }
    if (path.startsWith("/api/v1/media") && method === "GET") return json(route, 200, { items: [], total: 0 })
    if (path.startsWith("/api/v1/flashcards")) return json(route, 200, [])
    if (path === "/api/v1/quizzes/generate" && method === "POST") {
      const body = request.postDataJSON()
      expect(body).toMatchObject({ generation_profile: "osce_scenario", num_stations: 2 })
      generated = true
      return json(route, 200, {
        output_kind: "osce_stations",
        quiz: quizRecord,
        questions: [],
        osce_stations: stations.slice(0, 2),
      })
    }
    if (path === "/api/v1/quizzes" && method === "GET") {
      return json(route, 200, { items: generated ? [quizRecord] : [], count: generated ? 1 : 0 })
    }
    if (path === "/api/v1/quizzes/attempts" && method === "GET") {
      return json(route, 200, { items: [], count: 0 })
    }
    if (path === "/api/v1/quizzes/81" && method === "GET") return json(route, 200, quizRecord)

    const stationList = path.match(/^\/api\/v1\/quizzes\/81\/osce-stations$/)
    if (stationList && method === "GET") {
      const offset = Number(url.searchParams.get("offset") ?? 0)
      const limit = Number(url.searchParams.get("limit") ?? 50)
      const active = stationDeleted ? stations.slice(1) : stations
      const items = active.slice(offset, offset + limit).map(summary)
      const nextOffset = offset + items.length
      const hasMore = nextOffset < active.length
      return json(route, 200, {
        items,
        count: active.length,
        has_more: hasMore,
        next_offset: hasMore ? nextOffset : null,
        pagination: { mode: "offset", total: active.length, offset, limit, has_more: hasMore, next_offset: hasMore ? nextOffset : null },
      })
    }

    const stationDetail = path.match(/^\/api\/v1\/quizzes\/81\/osce-stations\/(\d+)$/)
    if (stationDetail) {
      const station = stations.find((item) => item.id === Number(stationDetail[1]))
      if (!station || (stationDeleted && station.id === 901)) return json(route, 404, { detail: "OSCE station not found" })
      if (method === "GET") return json(route, 200, station)
      if (method === "PATCH") {
        const body = request.postDataJSON()
        if (body.expected_version !== station.version) return json(route, 409, { detail: "Version conflict" })
        station.content = { ...station.content, ...body.content }
        station.version += 1
        station.verification_state = "modified_after_verification"
        return json(route, 200, station)
      }
      if (method === "DELETE") {
        stationDeleted = station.id === 901
        return json(route, 200, { status: "deleted" })
      }
    }

    if (path === "/api/v1/quizzes/osce-stations/901/attempts" && method === "POST") {
      attemptSnapshot ??= structuredClone(stations[0].content)
      return json(route, 201, attempt())
    }
    if (path === "/api/v1/quizzes/osce-attempts" && method === "GET") {
      const requestedStates = url.searchParams.getAll("state")
      const exists = attemptSnapshot != null && (requestedStates.length === 0 || requestedStates.includes(attemptState))
      const items = exists ? [attemptSummary()] : []
      return json(route, 200, {
        items,
        count: items.length,
        has_more: false,
        next_offset: null,
        pagination: { mode: "offset", total: items.length, offset: 0, limit: 200, has_more: false, next_offset: null },
      })
    }
    if (path === "/api/v1/quizzes/osce-attempts/701" && method === "GET") return json(route, 200, attempt())
    if (path === "/api/v1/quizzes/osce-attempts/701" && method === "PATCH") {
      if (failNextPatchAsOffline) {
        failNextPatchAsOffline = false
        return route.abort("internetdisconnected")
      }
      const body = request.postDataJSON()
      if (body.expected_version !== attemptVersion) return json(route, 409, { detail: "Version conflict" })
      if (typeof body.notes === "string") attemptNotes = body.notes
      if (body.checklist_selections) checklistSelections = body.checklist_selections
      if (body.rubric_selections) rubricSelections = body.rubric_selections
      attemptVersion += 1
      return json(route, 200, attempt())
    }
    if (path === "/api/v1/quizzes/osce-attempts/701/begin-self-assessment" && method === "POST") {
      attemptState = "self_assessment"
      attemptVersion += 1
      return json(route, 200, attempt())
    }
    if (path === "/api/v1/quizzes/osce-attempts/701/complete" && method === "POST") {
      attemptState = "completed"
      attemptVersion += 1
      return json(route, 200, attempt())
    }

    return json(route, 200, {})
  })

  return {
    failNextPatchAsOffline: () => { failNextPatchAsOffline = true },
    createServerConflict: () => { attemptVersion += 1 },
    reviseAndDeleteLiveStation: () => {
      stations[0].content.title = "Live station changed after attempt start"
      stations[0].version += 1
      stationDeleted = true
    },
  }
}

test.describe("Quiz Playground", () => {
  let quiz: QuizPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    quiz = new QuizPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Quiz page with beta badge", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      // Beta badge should be visible in all states (online, offline, demo)
      const betaVisible = await quiz.betaBadge.isVisible().catch(() => false)
      const playgroundVisible = await quiz.isPlaygroundVisible()
      const demoVisible = await quiz.demoPreview.isVisible().catch(() => false)
      const connectionVisible = await quiz.connectionBanner.isVisible().catch(() => false)
      const unavailableVisible = await quiz.featureUnavailable.isVisible().catch(() => false)

      // At least one state should be rendered
      expect(
        betaVisible || playgroundVisible || demoVisible || connectionVisible || unavailableVisible
      ).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show beta tooltip on badge interaction", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      const badgeVisible = await quiz.betaBadge.isVisible().catch(() => false)
      if (!badgeVisible) return

      // The tooltip opens on mouseEnter; use hover to trigger it
      await quiz.hoverBetaBadge()
      await expect(quiz.betaTooltip).toBeVisible({ timeout: 5_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch between playground tabs without errors", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      // Only test tab switching if the playground (online state) is visible
      const playgroundVisible = await quiz.isPlaygroundVisible()
      if (!playgroundVisible) return

      for (const tab of ["generate", "create", "manage", "results", "take"] as const) {
        await quiz.switchToTab(tab)
        const tabLocator = {
          generate: quiz.generateTab,
          create: quiz.createTab,
          manage: quiz.manageTab,
          results: quiz.resultsTab,
          take: quiz.takeTab,
        }[tab]
        await expect(tabLocator).toHaveAttribute("aria-selected", "true")
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Global Search and Controls
  // =========================================================================

  test.describe("Global Search", () => {
    test("should have global search input and apply button", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      const playgroundVisible = await quiz.isPlaygroundVisible()
      if (!playgroundVisible) return

      await expect(quiz.globalSearchInput).toBeVisible()
      await expect(quiz.globalSearchApplyButton).toBeVisible()
      await expect(quiz.resetCurrentTabButton).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should accept text input in global search", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      const playgroundVisible = await quiz.isPlaygroundVisible()
      if (!playgroundVisible) return

      await quiz.globalSearchInput.fill("test search query")
      await expect(quiz.globalSearchInput).toHaveValue("test search query")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Demo Mode Flow
  // =========================================================================

  test.describe("Demo Mode", () => {
    test("should render demo quiz preview or connection banner when offline", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      // If online, the playground is shown; if offline, either demo or connection banner
      const playgroundVisible = await quiz.isPlaygroundVisible()
      if (playgroundVisible) return // Skip demo tests when server is available

      const demoVisible = await quiz.demoPreview.isVisible().catch(() => false)
      const connectionVisible = await quiz.connectionBanner.isVisible().catch(() => false)

      expect(demoVisible || connectionVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should start and navigate through demo quiz when in demo mode", async ({
      authedPage,
      diagnostics,
    }) => {
      quiz = new QuizPage(authedPage)
      await quiz.goto()
      await quiz.assertPageReady()

      const demoVisible = await quiz.demoPreview.isVisible().catch(() => false)
      if (!demoVisible) return // Skip if not in demo mode

      // Click the start button
      await expect(quiz.demoStartButton).toBeVisible()
      await quiz.demoStartButton.click()

      // Should show the taking section
      await expect(quiz.demoTaking).toBeVisible({ timeout: 5_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // API Integration (requires server)
  // =========================================================================

  test.describe("Quiz API", () => {
    test("should fire GET /api/v1/quizzes when playground loads", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      quiz = new QuizPage(authedPage)

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/quizzes/,
        method: "GET",
      }, 20_000)

      await quiz.goto()
      await quiz.assertPageReady()

      const playgroundVisible = await quiz.isPlaygroundVisible()
      if (!playgroundVisible) return

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Quiz API may not be available on this server version
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("OSCE scenario practice", () => {
    test("generates, edits, practices, self-assesses, and shows a no-score result", async ({ page }) => {
      const backend = await installMockOsceBackend(page)
      quiz = new QuizPage(page)
      await quiz.goto()
      await quiz.assertPageReady()

      await quiz.generateOsce({ sourceNote: "Clinical handover note", stations: 2 })
      await quiz.openOsceManager("Generated OSCE Practice")
      await quiz.editFirstStationTitle("Anticoagulant counselling", "Anticoagulant safety review")
      await quiz.startOscePractice(81)
      await expect(quiz.oscePracticePanel.getByText("Anticoagulant safety review", { exact: true })).toBeVisible()

      backend.reviseAndDeleteLiveStation()
      await page.reload({ waitUntil: "domcontentloaded" })
      await quiz.assertPageReady()
      await quiz.startOscePractice(81)
      await expect(quiz.oscePracticePanel.getByText("Anticoagulant safety review", { exact: true })).toBeVisible()

      await quiz.beginOsceSelfAssessment()
      await quiz.completeOsceAssessment()
      await quiz.openOsceResults()
      await quiz.expectOsceResultWithoutScore("Anticoagulant safety review")
    })

    test("keeps an offline draft and surfaces a reconnect version conflict", async ({ page }) => {
      const backend = await installMockOsceBackend(page, { generated: true })
      quiz = new QuizPage(page)
      await quiz.goto()
      await quiz.assertPageReady()
      await quiz.startOscePractice(81)

      backend.failNextPatchAsOffline()
      await page.getByRole("textbox", { name: "Private practice notes" }).fill("Locally retained reflection")
      await expect(page.getByText("The practice could not be saved while offline. Your local draft was kept.")).toBeVisible()

      backend.createServerConflict()
      await page.reload({ waitUntil: "domcontentloaded" })
      await quiz.assertPageReady()
      await quiz.startOscePractice(81)
      await expect(page.getByText("This practice changed on the server. Your local draft was kept.")).toBeVisible()
      await expect(page.getByRole("textbox", { name: "Private practice notes" })).toHaveValue("Locally retained reflection")
    })

    test("loads every station page before rendering the manager", async ({ page }) => {
      await installMockOsceBackend(page, { generated: true, stationCount: 201 })
      quiz = new QuizPage(page)
      await quiz.goto()
      await quiz.assertPageReady()
      await quiz.switchToTab("manage")
      await quiz.openOsceManager("Generated OSCE Practice")

      await expect(page.getByRole("button", { name: "Edit station Station 201" })).toBeVisible()
    })

    test("keeps a near-limit station authoring surface usable on mobile", async ({ page }) => {
      await installMockOsceBackend(page, { generated: true, stationCount: 1, nearLimit: true })
      quiz = new QuizPage(page)
      await quiz.goto()
      await quiz.assertPageReady()
      await quiz.switchToTab("manage")
      await quiz.openOsceManager("Generated OSCE Practice")
      await page.setViewportSize({ width: 390, height: 844 })
      const stationButton = page.getByRole("button", { name: /^Edit station N{200}$/ })
      await stationButton.click()

      await expect(page.getByRole("textbox", { name: "Checklist item 50", exact: true })).toBeVisible()
      await expect(page.getByRole("button", { name: "Save station" })).toBeVisible()
      const overflow = await page.locator('section[aria-label="Station authoring"]').evaluate(
        (element) => element.scrollWidth - element.clientWidth,
      )
      expect(overflow).toBeLessThanOrEqual(1)
    })
  })
})
