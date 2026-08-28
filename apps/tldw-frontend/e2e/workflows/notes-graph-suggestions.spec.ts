import { mkdirSync } from "node:fs"
import path from "node:path"

import type { Page, Route, TestInfo } from "@playwright/test"

import { expect, test } from "../utils/fixtures"

const NOW = "2026-08-28T12:00:00Z"
const SOURCE_NOTE_ID = "source-note"
const SOURCE_FINGERPRINT = `sha256:${"a".repeat(64)}`
const TARGET_FINGERPRINT = `sha256:${"b".repeat(64)}`
const CAPABILITY_REVISION = `sha256:${"c".repeat(64)}`
const ENDPOINT_REVISION = `sha256:${"d".repeat(64)}`
const LONG_PROVIDER =
  "Deterministic Local Provider With A Deliberately Long Disclosure Name"
const LONG_MODEL =
  "notes-grounding-model-with-an-intentionally-long-version-and-context-label"
const LONG_TARGET =
  "A grounded related note with a deliberately long title that must wrap without covering adjacent review controls"
const LONG_TAG =
  "Systems Thinking Across Distributed Knowledge Workflows And Durable Evidence"
const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const SUGGESTION_BASE = `/api/v1/notes/${SOURCE_NOTE_ID}/graph/suggestions`

const note = {
  id: SOURCE_NOTE_ID,
  title: "Second brain source note",
  content: "Evidence about durable synchronization and grounded graph review.",
  version: 3,
  keywords: [],
  created_at: NOW,
  updated_at: NOW
}

const STATIC_NOTE_RESPONSES: Array<[string, unknown]> = [
  ["/api/v1/notes/keywords", { keywords: [], total: 0 }],
  ["/api/v1/notes/collections", { collections: [], total: 0 }],
  ["/api/v1/notes/moodboards", { moodboards: [], total: 0 }],
  ["/api/v1/notes/trash", { notes: [], total: 0 }]
]

const graphNode = (
  id: string,
  label: string,
  type: "note" | "tag" | "source" = "note"
) => ({
  id,
  type,
  label,
  created_at: type === "note" ? NOW : null,
  deleted: false,
  degree: type === "note" ? 1 : null,
  tag_count: type === "note" ? 0 : null,
  primary_source_id: null
})

const relatedSuggestion = {
  id: "related-suggestion",
  run_id: "run-1",
  kind: "related_note",
  state: "pending",
  revision: 1,
  source_note_id: SOURCE_NOTE_ID,
  source_fingerprint: SOURCE_FINGERPRINT,
  target_note_id: "suggested-target",
  target_fingerprint: TARGET_FINGERPRINT,
  target_title: LONG_TARGET,
  normalized_tag: null,
  display_tag: null,
  existing_tag: false,
  match_strength: "strong",
  rationale:
    "Both notes describe durable synchronization boundaries, explicit review, and evidence-backed recovery across interrupted knowledge workflows.",
  evidence: [
    {
      side: "source",
      note_id: SOURCE_NOTE_ID,
      field: "content",
      start_offset: 0,
      end_offset: 42,
      text: "Evidence about durable synchronization and grounded graph review."
    },
    {
      side: "target",
      note_id: "suggested-target",
      field: "content",
      start_offset: 0,
      end_offset: 40,
      text: "Durable synchronization requires explicit evidence review."
    }
  ],
  updated_at: NOW
}

const tagSuggestion = {
  id: "tag-suggestion",
  run_id: "run-1",
  kind: "tag",
  state: "pending",
  revision: 1,
  source_note_id: SOURCE_NOTE_ID,
  source_fingerprint: SOURCE_FINGERPRINT,
  target_note_id: null,
  target_fingerprint: null,
  target_title: null,
  normalized_tag: "systems-thinking",
  display_tag: LONG_TAG,
  existing_tag: true,
  match_strength: "possible",
  rationale:
    "The selected note applies systems thinking to synchronization, recovery, review, and the boundaries between authoritative and provisional knowledge.",
  evidence: [
    {
      side: "source",
      note_id: SOURCE_NOTE_ID,
      field: "content",
      start_offset: 0,
      end_offset: 42,
      text: "Evidence about durable synchronization and grounded graph review."
    }
  ],
  updated_at: NOW
}

type TerminalState = "succeeded" | "stale" | "failed"
type RunState = "queued" | "running" | "publishing" | TerminalState
type SuggestionCall = {
  method: string
  path: string
  commandUuid: string | null
}

class NotesGraphFixture {
  readonly calls: SuggestionCall[] = []
  readonly graphCalls: string[] = []
  readonly outcomes: TerminalState[]
  readonly authorized: boolean
  accepted = false
  rejected = false
  reset = false
  published = false
  private runs: Array<{
    id: string
    outcome: TerminalState
    state: RunState
    cancelled: boolean
    servedState: RunState | "cancelled" | null
    releaseDetail: (() => void) | null
  }> = []

  constructor(options: {
    authorized?: boolean
    outcomes?: TerminalState[]
  } = {}) {
    this.authorized = options.authorized ?? true
    this.outcomes = options.outcomes ?? ["succeeded"]
  }

  get admissionCalls() {
    return this.calls.filter(
      (call) => call.method === "POST" && call.path.endsWith("/runs")
    )
  }

  setRunState(runId: string, state: RunState) {
    const record = this.runs.find((item) => item.id === runId)
    if (!record) throw new Error(`Unknown run ${runId}`)
    const sequence: RunState[] = ["queued", "running", "publishing", record.outcome]
    if (!sequence.includes(state)) {
      throw new Error(`Invalid ${runId} fixture transition to ${state}`)
    }
    record.state = state
    if (state === "succeeded") this.published = true
    record.releaseDetail?.()
    record.releaseDetail = null
  }

  count(method: string, pathSuffix: string) {
    return this.calls.filter(
      (call) => call.method === method && call.path.endsWith(pathSuffix)
    ).length
  }

  private graph() {
    const fixedNodes = [
      graphNode(`note:${SOURCE_NOTE_ID}`, note.title),
      graphNode("source:publication", "Web source with a long canonical label", "source"),
      graphNode("tag:durable-review", "Durable review", "tag"),
      graphNode("note:suggested-target", LONG_TARGET)
    ]
    const numberedNodes = Array.from({ length: 99 }, (_, index) =>
      graphNode(
        `note:relationship-${String(index + 1).padStart(3, "0")}`,
        `Relationship ${String(index + 1).padStart(3, "0")} with a label that remains readable at narrow widths`
      )
    )
    const fixedEdges = [
      {
        id: "source-membership",
        source: `note:${SOURCE_NOTE_ID}`,
        target: "source:publication",
        type: "source_membership",
        directed: false,
        weight: 1,
        label: null
      },
      {
        id: "tag-membership",
        source: `note:${SOURCE_NOTE_ID}`,
        target: "tag:durable-review",
        type: "tag_membership",
        directed: false,
        weight: 1,
        label: null
      }
    ]
    const numberedEdges = numberedNodes.map((node, index) => ({
      id: `manual-${String(index + 1).padStart(3, "0")}`,
      source: `note:${SOURCE_NOTE_ID}`,
      target: node.id,
      type: "manual",
      directed: false,
      weight: 1,
      label: null
    }))
    const acceptedEdge = this.accepted
      ? [
          {
            id: "accepted-related-suggestion",
            source: `note:${SOURCE_NOTE_ID}`,
            target: "note:suggested-target",
            type: "manual",
            directed: false,
            weight: 1,
            label: null
          }
        ]
      : []
    return {
      nodes: [...fixedNodes, ...numberedNodes],
      edges: [...fixedEdges, ...numberedEdges, ...acceptedEdge],
      truncated: false,
      truncated_by: [],
      has_more: false,
      cursor: null,
      limits: { max_nodes: 120, max_edges: 480, max_degree: 300 },
      radius_cap_applied: false,
      active_note_count: 9,
      all_notes_note_cap: 8,
      all_notes_eligible: false,
      suggestions_authorized: this.authorized
    }
  }

  private capabilities() {
    return {
      provider: LONG_PROVIDER,
      model: LONG_MODEL,
      endpoint_origin_revision: ENDPOINT_REVISION,
      data_boundary: "remote",
      disclosure_external: true,
      outbound_data_categories: [
        "selected_note_title",
        "selected_note_excerpts",
        "candidate_note_titles",
        "candidate_note_excerpts",
        "existing_tag_labels"
      ],
      generation_available: true,
      unavailable_reason: null,
      limits: {
        max_candidates: 30,
        max_relationships: 5,
        max_tags: 5,
        max_new_tags: 2,
        max_tag_catalog: 100,
        max_estimated_input_tokens: 24_000,
        max_output_tokens: 2_000,
        provider_timeout_seconds: 120,
        response_candidates: 1
      },
      allowed_actions: [
        "generate",
        "cancel",
        "accept",
        "reject",
        "reset_rejections"
      ],
      revision: CAPABILITY_REVISION
    }
  }

  private run(
    record: { id: string; outcome: TerminalState; cancelled: boolean },
    state: string,
    revision: number
  ) {
    const terminal = ["succeeded", "failed", "cancelled", "stale"].includes(state)
    return {
      id: record.id,
      provider: LONG_PROVIDER,
      model: LONG_MODEL,
      state,
      revision,
      created_at: NOW,
      started_at: state === "queued" ? null : NOW,
      completed_at: terminal ? NOW : null,
      suggestion_count: state === "succeeded" ? 2 : 0,
      related_note_count: state === "succeeded" ? 1 : 0,
      tag_count: state === "succeeded" ? 1 : 0,
      invalid_item_count: 0,
      cancellation_available: ["queued", "running"].includes(state),
      error_code:
        state === "failed"
          ? "notes_graph_provider_unavailable"
          : state === "stale"
            ? "notes_graph_source_changed"
            : null,
      guidance_key:
        state === "failed"
          ? "retry_generation"
          : state === "stale"
            ? "refresh_note"
            : null
    }
  }

  private async nextRunResponse(record: (typeof this.runs)[number]) {
    let state: RunState | "cancelled" = record.cancelled ? "cancelled" : record.state
    if (record.servedState === state) {
      await new Promise<void>((resolve) => {
        record.releaseDetail = resolve
      })
      state = record.cancelled ? "cancelled" : record.state
    }
    record.servedState = state
    const revision = ["queued", "running", "publishing"].indexOf(state) + 1
    return this.run(record, state, revision > 0 ? revision : 4)
  }

  private suggestions() {
    const items = this.published
      ? [
          ...(this.accepted ? [] : [relatedSuggestion]),
          ...(this.rejected && !this.reset ? [] : [tagSuggestion])
        ]
      : []
    return {
      items,
      next_cursor: null,
      current_source_fingerprint: SOURCE_FINGERPRINT,
      rejection_set_revision: this.rejected && !this.reset ? 1 : 0,
      rejection_count: this.rejected && !this.reset ? 1 : 0
    }
  }

  private async fulfill(route: Route, body: unknown, headers = {}) {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      headers: {
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "*",
        "access-control-expose-headers": "etag",
        ...headers
      },
      body: JSON.stringify(body)
    })
  }

  async handle(route: Route) {
    const request = route.request()
    const url = new URL(request.url())
    const method = request.method().toUpperCase()
    const requestPath = url.pathname

    if (requestPath === "/api/v1/notes/graph" && method === "GET") {
      this.graphCalls.push(url.toString())
      await this.fulfill(route, this.graph())
      return
    }

    if (requestPath.includes("/graph/suggestions")) {
      const commandUuid = request.headers()["idempotency-key"] ?? null
      this.calls.push({ method, path: requestPath, commandUuid })

      if (requestPath.endsWith("/capabilities") && method === "GET") {
        await this.fulfill(route, this.capabilities(), {
          etag: `"${CAPABILITY_REVISION}"`
        })
        return
      }
      if (requestPath.endsWith("/runs") && method === "POST") {
        const outcome = this.outcomes[this.runs.length] ?? "succeeded"
        const record = {
          id: `run-${this.runs.length + 1}`,
          outcome,
          state: "queued" as const,
          cancelled: false,
          servedState: null,
          releaseDetail: null
        }
        this.runs.push(record)
        await this.fulfill(route, this.run(record, "queued", 1))
        return
      }
      if (requestPath.endsWith("/runs") && method === "GET") {
        const active = this.runs.filter((record) => {
          const terminal = record.cancelled || record.state === record.outcome
          return !terminal
        })
        await this.fulfill(route, {
          items: active.map((record) => this.run(record, record.state, 1)),
          next_cursor: null
        })
        return
      }
      const runMatch = requestPath.match(/\/runs\/(run-\d+)$/)
      if (runMatch && method === "GET") {
        const record = this.runs.find((item) => item.id === runMatch[1])
        if (!record) throw new Error(`Unknown run ${runMatch[1]}`)
        await this.fulfill(route, await this.nextRunResponse(record))
        return
      }
      const cancelMatch = requestPath.match(/\/runs\/(run-\d+)\/cancel$/)
      if (cancelMatch && method === "POST") {
        const record = this.runs.find((item) => item.id === cancelMatch[1])
        if (!record) throw new Error(`Unknown run ${cancelMatch[1]}`)
        record.cancelled = true
        record.releaseDetail?.()
        record.releaseDetail = null
        await this.fulfill(route, {
          resource_id: record.id,
          state: "cancelling",
          revision: 2,
          cleared_count: null
        })
        return
      }
      if (requestPath.endsWith("/suggestions") && method === "GET") {
        await this.fulfill(route, this.suggestions())
        return
      }
      if (requestPath.endsWith("/related-suggestion/accept") && method === "POST") {
        this.accepted = true
        await this.fulfill(route, {
          resource_id: "related-suggestion",
          state: "accepted",
          revision: 2,
          cleared_count: null
        })
        return
      }
      if (requestPath.endsWith("/tag-suggestion/reject") && method === "POST") {
        this.rejected = true
        this.reset = false
        await this.fulfill(route, {
          resource_id: "tag-suggestion",
          state: "rejected",
          revision: 2,
          cleared_count: null
        })
        return
      }
      if (requestPath.endsWith("/rejections/reset") && method === "POST") {
        this.reset = true
        await this.fulfill(route, {
          resource_id: SOURCE_NOTE_ID,
          state: "reset",
          revision: 2,
          cleared_count: 1
        })
        return
      }
      throw new Error(`Unhandled suggestion request: ${method} ${requestPath}`)
    }

    if (requestPath.startsWith("/api/v1/notes/title-settings")) {
      await this.fulfill(route, {
        llm_enabled: false,
        default_strategy: "heuristic"
      })
      return
    }
    if (requestPath.startsWith("/api/v1/notes/search")) {
      await this.fulfill(route, { notes: [note], total: 1 })
      return
    }
    if (requestPath === "/api/v1/notes" || requestPath === "/api/v1/notes/") {
      await this.fulfill(route, {
        items: [note],
        pagination: { total_items: 1 }
      })
      return
    }
    if (requestPath === `/api/v1/notes/${SOURCE_NOTE_ID}`) {
      await this.fulfill(route, { ...note, links: [] })
      return
    }
    if (requestPath.includes("/neighbors")) {
      await this.fulfill(route, { nodes: [], edges: [] })
      return
    }
    const staticResponse = STATIC_NOTE_RESPONSES.find(([prefix]) =>
      requestPath.startsWith(prefix)
    )
    if (staticResponse) {
      await this.fulfill(route, staticResponse[1])
      return
    }
    await route.continue()
  }
}

const installFixture = async (page: Page, fixture: NotesGraphFixture) => {
  await page.route("**/api/v1/**", (route) => fixture.handle(route))
}

const openGraph = async (page: Page) => {
  await page.goto("/notes", { waitUntil: "domcontentloaded" })
  const graphEntry = page.getByTestId("notes-view-mode-graph")
  await expect(graphEntry).toBeVisible()
  await graphEntry.click()
  await expect(page.getByTestId("notes-graph-workspace")).toBeVisible()
  await expect(page.getByTestId("notes-graph-canvas")).toBeVisible()
}

const openSuggestions = async (page: Page) => {
  const tab = page.getByRole("tab", { name: "Suggestions" })
  await expect(tab).toBeVisible()
  await tab.click()
  await expect(tab).toHaveAttribute("aria-selected", "true")
}

const closeMobileNotesList = async (page: Page) => {
  const backdrop = page.getByTestId("notes-mobile-sidebar-backdrop")
  if ((await backdrop.count()) > 0 && (await backdrop.isVisible())) {
    await backdrop.click()
  }
  const list = page.getByTestId("notes-list-region")
  await expect(list).toHaveClass(/-translate-x-full/)
  await expect
    .poll(async () => {
      const bounds = await list.boundingBox()
      return bounds ? bounds.x + bounds.width : 0
    })
    .toBeLessThanOrEqual(1)
  await expect(backdrop).toHaveCount(0)
}

const closeDesktopNotesList = async (page: Page) => {
  const toggle = page.getByTestId("notes-desktop-sidebar-toggle")
  await expect(toggle).toBeVisible()
  if ((await toggle.getAttribute("aria-label")) === "Collapse sidebar") {
    await toggle.click()
  }
  await expect(toggle).toHaveAttribute("aria-label", "Expand sidebar")
  const list = page.getByTestId("notes-list-region")
  await expect(list).toHaveClass(/w-0/)
  await expect.poll(async () => (await list.boundingBox())?.width ?? 0).toBeLessThanOrEqual(1)
}

const advanceRun = async (
  page: Page,
  fixture: NotesGraphFixture,
  runId: string,
  terminal: TerminalState
) => {
  const terminalLabels: Record<TerminalState, string> = {
    succeeded: "Succeeded",
    stale: "Stale",
    failed: "Failed"
  }
  const states: Array<[RunState, string]> = [
    ["running", "Running"],
    ["publishing", "Publishing"],
    [terminal, terminalLabels[terminal]]
  ]
  for (const [state, label] of states) {
    fixture.setRunState(runId, state)
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      label,
      { timeout: state === terminal ? 15_000 : 10_000 }
    )
  }
}

const canvasPixels = async (page: Page) =>
  page.getByTestId("notes-graph-canvas").evaluate((root) => {
    const canvases = Array.from(root.querySelectorAll("canvas"))
    const colors = new Set<string>()
    let paintedPixels = 0
    for (const canvas of canvases) {
      const context = canvas.getContext("2d")
      if (!context || canvas.width === 0 || canvas.height === 0) continue
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
      const stride = Math.max(1, Math.floor(Math.sqrt(pixels.length / 400_000)))
      for (let offset = 0; offset < pixels.length; offset += 4 * stride) {
        if (pixels[offset + 3] === 0) continue
        paintedPixels += 1
        colors.add(`${pixels[offset]}:${pixels[offset + 1]}:${pixels[offset + 2]}`)
      }
    }
    return {
      canvasCount: canvases.length,
      paintedPixels,
      distinctColors: colors.size
    }
  })

const geometry = async (page: Page) =>
  page.getByTestId("notes-graph-workspace").evaluate((workspace) => {
    const box = (element: Element) => {
      const rect = element.getBoundingClientRect()
      return {
        left: rect.left,
        right: rect.right,
        top: rect.top,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height
      }
    }
    const primary = workspace.querySelector('[data-testid="notes-graph-primary-view"]')
    const toolbar = workspace.querySelector('[data-testid="notes-graph-toolbar"]')
    const inspector = workspace.querySelector(
      '[data-testid="notes-graph-inspector-region"]'
    )
    const visualViewport = window.visualViewport
    const controls = Array.from(
      workspace.querySelectorAll("button:not([hidden]), input:not([hidden]), select:not([hidden])")
    ).filter((element) => {
      const rect = element.getBoundingClientRect()
      const style = getComputedStyle(element)
      return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden"
    })
    const overlapPairs: string[] = []
    for (let leftIndex = 0; leftIndex < controls.length; leftIndex += 1) {
      const left = controls[leftIndex].getBoundingClientRect()
      for (let rightIndex = leftIndex + 1; rightIndex < controls.length; rightIndex += 1) {
        const right = controls[rightIndex].getBoundingClientRect()
        const overlapWidth = Math.min(left.right, right.right) - Math.max(left.left, right.left)
        const overlapHeight = Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top)
        if (overlapWidth > 2 && overlapHeight > 2) {
          overlapPairs.push(`${leftIndex}:${rightIndex}`)
        }
      }
    }
    return {
      viewportWidth: innerWidth,
      visualViewport: visualViewport
        ? {
            width: visualViewport.width,
            height: visualViewport.height,
            scale: visualViewport.scale,
            offsetLeft: visualViewport.offsetLeft,
            offsetTop: visualViewport.offsetTop,
            right: visualViewport.offsetLeft + visualViewport.width,
            bottom: visualViewport.offsetTop + visualViewport.height
          }
        : null,
      horizontalOverflow: document.documentElement.scrollWidth - innerWidth,
      workspace: box(workspace),
      toolbar: toolbar ? box(toolbar) : null,
      primary: primary ? box(primary) : null,
      inspector: inspector ? box(inspector) : null,
      controlBoxes: controls.map(box),
      toolbarControlBoxes: controls.filter((element) => toolbar?.contains(element)).map(box),
      visibleOverlayCount: Array.from(
        document.querySelectorAll(
          '[role="dialog"], [data-testid="notes-mobile-sidebar-backdrop"], .ant-drawer-mask'
        )
      ).filter((element) => {
        const rect = element.getBoundingClientRect()
        const style = getComputedStyle(element)
        return (
          rect.width > 0 &&
          rect.height > 0 &&
          style.visibility !== "hidden" &&
          style.display !== "none" &&
          Number(style.opacity || "1") > 0
        )
      }).length,
      overlapPairs
    }
  })

const assertVisualContract = async (page: Page) => {
  const report = await geometry(page)
  expect(report.horizontalOverflow, JSON.stringify(report)).toBeLessThanOrEqual(1)
  expect(report.workspace.left, JSON.stringify(report)).toBeGreaterThanOrEqual(-1)
  expect(report.workspace.right, JSON.stringify(report)).toBeLessThanOrEqual(
    report.viewportWidth + 1
  )
  expect(report.visualViewport, JSON.stringify(report)).not.toBeNull()
  expect(report.workspace.left, JSON.stringify(report)).toBeGreaterThanOrEqual(
    (report.visualViewport?.offsetLeft ?? 0) - 1
  )
  expect(report.workspace.right, JSON.stringify(report)).toBeLessThanOrEqual(
    (report.visualViewport?.right ?? report.viewportWidth) + 1
  )
  expect(report.toolbar?.left, JSON.stringify(report)).toBeGreaterThanOrEqual(
    (report.visualViewport?.offsetLeft ?? 0) - 1
  )
  expect(report.toolbar?.right, JSON.stringify(report)).toBeLessThanOrEqual(
    (report.visualViewport?.right ?? report.viewportWidth) + 1
  )
  expect(report.inspector?.left, JSON.stringify(report)).toBeGreaterThanOrEqual(
    (report.visualViewport?.offsetLeft ?? 0) - 1
  )
  expect(report.inspector?.right, JSON.stringify(report)).toBeLessThanOrEqual(
    (report.visualViewport?.right ?? report.viewportWidth) + 1
  )
  for (const bounds of report.controlBoxes) {
    expect(bounds.left, JSON.stringify(report)).toBeGreaterThanOrEqual(
      (report.visualViewport?.offsetLeft ?? 0) - 1
    )
    expect(bounds.right, JSON.stringify(report)).toBeLessThanOrEqual(
      (report.visualViewport?.right ?? report.viewportWidth) + 1
    )
  }
  expect(report.primary?.width, JSON.stringify(report)).toBeGreaterThan(250)
  expect(report.inspector?.width, JSON.stringify(report)).toBeGreaterThan(250)
  expect(report.visibleOverlayCount, JSON.stringify(report)).toBe(0)
  expect(report.overlapPairs, JSON.stringify(report)).toEqual([])
  await expect(page.getByTestId("notes-graph-workspace").locator("[aria-live]"))
    .toHaveCount(1)
  await expect(page.getByRole("dialog")).toHaveCount(0)
  const pixels = await canvasPixels(page)
  expect(pixels.canvasCount, JSON.stringify(pixels)).toBeGreaterThan(0)
  expect(pixels.paintedPixels, JSON.stringify(pixels)).toBeGreaterThan(50)
  expect(pixels.distinctColors, JSON.stringify(pixels)).toBeGreaterThan(1)
  return { geometry: report, pixels }
}

const assertPageScaleOriginContract = async (page: Page) => {
  const report = await geometry(page)
  const viewport = report.visualViewport
  expect(viewport, JSON.stringify(report)).not.toBeNull()
  expect(viewport?.scale, JSON.stringify(report)).toBeCloseTo(2, 1)
  expect(viewport?.width, JSON.stringify(report)).toBeCloseTo(720, 0)
  expect(viewport?.offsetLeft, JSON.stringify(report)).toBeCloseTo(0, 0)
  expect(viewport?.offsetTop, JSON.stringify(report)).toBeCloseTo(0, 0)
  expect(report.horizontalOverflow, JSON.stringify(report)).toBeLessThanOrEqual(1)
  expect(report.workspace.left, JSON.stringify(report)).toBeGreaterThanOrEqual(-1)
  expect(report.workspace.right, JSON.stringify(report)).toBeLessThanOrEqual(
    report.viewportWidth + 1
  )
  for (const bounds of [report.toolbar, report.primary]) {
    expect(bounds, JSON.stringify(report)).not.toBeNull()
    expect(bounds?.left, JSON.stringify(report)).toBeGreaterThanOrEqual(
      (viewport?.offsetLeft ?? 0) - 1
    )
    expect(bounds?.left, JSON.stringify(report)).toBeLessThan(viewport?.right ?? 0)
    expect(
      Math.min(bounds?.right ?? 0, viewport?.right ?? 0) -
        Math.max(bounds?.left ?? 0, viewport?.offsetLeft ?? 0),
      JSON.stringify(report)
    ).toBeGreaterThan(250)
  }
  const visibleToolbarControls = report.toolbarControlBoxes.filter(
    (bounds) =>
      bounds.right > (viewport?.offsetLeft ?? 0) &&
      bounds.left < (viewport?.right ?? 0)
  )
  expect(visibleToolbarControls.length, JSON.stringify(report)).toBeGreaterThan(0)
  const fullyContainedToolbarControls = visibleToolbarControls.filter(
    (bounds) =>
      bounds.left >= (viewport?.offsetLeft ?? 0) - 1 &&
      bounds.right <= (viewport?.right ?? 0) + 1
  )
  expect(fullyContainedToolbarControls.length, JSON.stringify(report)).toBeGreaterThanOrEqual(4)
  for (const bounds of visibleToolbarControls) {
    const visibleLeft = Math.max(bounds.left, viewport?.offsetLeft ?? 0)
    const visibleRight = Math.min(bounds.right, viewport?.right ?? 0)
    expect(visibleLeft, JSON.stringify(report)).toBeGreaterThanOrEqual(
      viewport?.offsetLeft ?? 0
    )
    expect(visibleRight, JSON.stringify(report)).toBeLessThanOrEqual(
      viewport?.right ?? 0
    )
    expect(visibleRight - visibleLeft, JSON.stringify(report)).toBeGreaterThan(0)
  }
  expect(report.inspector?.left, JSON.stringify(report)).toBeGreaterThanOrEqual(
    (viewport?.right ?? 0) - 1
  )
  expect(report.inspector?.left, JSON.stringify(report)).toBeGreaterThanOrEqual(-1)
  expect(report.inspector?.right, JSON.stringify(report)).toBeLessThanOrEqual(
    report.viewportWidth + 1
  )
  expect(report.visibleOverlayCount, JSON.stringify(report)).toBe(0)
  expect(report.overlapPairs, JSON.stringify(report)).toEqual([])
  const pixels = await canvasPixels(page)
  expect(pixels.canvasCount, JSON.stringify(pixels)).toBeGreaterThan(0)
  expect(pixels.paintedPixels, JSON.stringify(pixels)).toBeGreaterThan(50)
  expect(pixels.distinctColors, JSON.stringify(pixels)).toBeGreaterThan(1)
  return { geometry: report, pixels }
}

const bringResponsiveInspectorIntoView = async (page: Page) => {
  await page
    .getByTestId("notes-graph-inspector-region")
    .evaluate((element) => element.scrollIntoView({ block: "center" }))
  const report = await geometry(page)
  expect(report.inspector?.top, JSON.stringify(report)).toBeGreaterThanOrEqual(
    (report.visualViewport?.offsetTop ?? 0) - 1
  )
  expect(report.inspector?.bottom, JSON.stringify(report)).toBeLessThanOrEqual(
    (report.visualViewport?.bottom ?? 0) + 1
  )
  expect(report.visibleOverlayCount, JSON.stringify(report)).toBe(0)
  expect(report.overlapPairs, JSON.stringify(report)).toEqual([])
  return report
}

const screenshot = async (page: Page, name: string) => {
  const directory = path.resolve(process.cwd(), "test-results/notes-graph-suggestions")
  mkdirSync(directory, { recursive: true })
  const output = path.join(directory, `${name}.png`)
  await page.screenshot({ path: output, fullPage: false })
  return output
}

const requestMultiset = (calls: SuggestionCall[]) =>
  Object.fromEntries(
    [...calls]
      .map((call) => `${call.method} ${call.path}`)
      .sort()
      .reduce<Array<[string, number]>>((entries, key) => {
        const previous = entries.at(-1)
        if (previous?.[0] === key) previous[1] += 1
        else entries.push([key, 1])
        return entries
      }, [])
  )

const assertExactSuggestionRequests = (
  calls: SuggestionCall[],
  expected: Record<string, number>
) => {
  expect(requestMultiset(calls)).toEqual(expected)
  const commands = calls.filter((call) => call.method !== "GET")
  expect(commands.every((call) => Boolean(call.commandUuid && UUID.test(call.commandUuid)))).toBe(
    true
  )
  expect(new Set(commands.map((call) => call.commandUuid)).size).toBe(commands.length)
}

const attachEvidence = async (
  testInfo: TestInfo,
  fixture: NotesGraphFixture,
  visualEvidence: unknown
) => {
  await testInfo.attach("notes-graph-request-and-visual-evidence.json", {
    body: JSON.stringify(
      {
        suggestionRequests: fixture.calls,
        graphRequests: fixture.graphCalls,
        visualEvidence
      },
      null,
      2
    ),
    contentType: "application/json"
  })
}

test.describe("Notes graph suggestions", () => {
  test("recovers one run, reviews grounded suggestions, and keeps responsive canvas geometry", async ({
    authedPage: page
  }, testInfo) => {
    const fixture = new NotesGraphFixture()
    await installFixture(page, fixture)
    await page.setViewportSize({ width: 1440, height: 1000 })
    await openGraph(page)

    await expect(page.getByRole("button", { name: "All notes" })).toBeDisabled()
    await expect(page.getByTestId("notes-graph-all-disabled-reason")).toContainText(
      "up to 8 active notes"
    )
    await openSuggestions(page)
    await page.getByRole("tab", { name: "Details" }).click()
    await expect(page.getByRole("heading", { name: note.title, exact: true })).toBeVisible()
    await openSuggestions(page)
    await expect(page.getByText(LONG_PROVIDER, { exact: true })).toBeVisible()
    await expect(page.getByText(LONG_MODEL, { exact: true })).toBeVisible()
    await expect(page.getByText("External", { exact: true })).toBeVisible()
    await expect(page.getByText("Selected note excerpts", { exact: true })).toBeVisible()

    await page.getByRole("button", { name: "Generate", exact: true }).click()
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Queued"
    )
    await expect.poll(() => fixture.count("GET", "/runs/run-1")).toBeGreaterThanOrEqual(1)
    expect(fixture.admissionCalls).toHaveLength(1)

    await page.reload({ waitUntil: "domcontentloaded" })
    const graphEntry = page.getByTestId("notes-view-mode-graph")
    await expect(graphEntry).toBeVisible()
    await graphEntry.click()
    await openSuggestions(page)
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Queued"
    )
    await expect.poll(() => fixture.count("GET", "/runs/run-1")).toBeGreaterThanOrEqual(1)
    fixture.setRunState("run-1", "running")
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Running",
      { timeout: 10_000 }
    )
    fixture.setRunState("run-1", "publishing")
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Publishing",
      { timeout: 10_000 }
    )
    await expect(page.getByRole("button", { name: "Cancel generation" })).toBeDisabled()
    fixture.setRunState("run-1", "succeeded")
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Succeeded",
      { timeout: 15_000 }
    )
    await expect(page.getByText(LONG_TARGET, { exact: true })).toBeVisible()
    await expect(page.getByText(LONG_TAG, { exact: true })).toBeVisible()
    const relatedReview = page.locator(
      '[data-suggestion-review-row="related-suggestion"]'
    )
    await expect(relatedReview.getByText(relatedSuggestion.rationale, { exact: true })).toBeVisible()
    await expect(relatedReview.getByText(relatedSuggestion.evidence[0].text, { exact: true })).toBeVisible()
    await expect(relatedReview.getByText(relatedSuggestion.evidence[1].text, { exact: true })).toBeVisible()
    expect(fixture.admissionCalls).toHaveLength(1)

    const relationshipsMode = page.getByRole("button", {
      name: "Relationships",
      exact: true
    })
    await relationshipsMode.focus()
    await relationshipsMode.press("Enter")
    await expect(page.getByTestId("notes-graph-relationships-view")).toBeVisible()
    const detailsTab = page.getByRole("tab", { name: "Details" })
    const suggestionsTab = page.getByRole("tab", { name: "Suggestions" })
    await detailsTab.focus()
    await detailsTab.press("End")
    await expect(suggestionsTab).toBeFocused()
    await expect(suggestionsTab).toHaveAttribute("aria-selected", "true")
    await suggestionsTab.press("Home")
    await expect(detailsTab).toBeFocused()
    await expect(detailsTab).toHaveAttribute("aria-selected", "true")
    await detailsTab.press("ArrowRight")
    await expect(suggestionsTab).toBeFocused()
    await expect(suggestionsTab).toHaveAttribute("aria-selected", "true")
    await suggestionsTab.press("ArrowLeft")
    await expect(detailsTab).toBeFocused()
    await expect(detailsTab).toHaveAttribute("aria-selected", "true")

    const visualEvidence: Record<string, unknown> = {}
    await page.getByRole("button", { name: "Canvas", exact: true }).click()
    visualEvidence.desktop = await assertVisualContract(page)
    visualEvidence.desktopScreenshot = await screenshot(page, "desktop")

    await page.getByRole("button", { name: "Collapse sidebar" }).click()
    await expect(page.getByRole("button", { name: "Expand sidebar" }).last()).toBeVisible()
    await page.setViewportSize({ width: 320, height: 900 })
    await closeMobileNotesList(page)
    visualEvidence.mobile320 = await assertVisualContract(page)
    visualEvidence.mobile320InspectorViewport = await bringResponsiveInspectorIntoView(page)
    visualEvidence.mobile320Screenshot = await screenshot(page, "mobile-320")

    await page.evaluate(() => window.scrollTo(0, 0))
    await page.setViewportSize({ width: 1440, height: 1000 })
    await closeDesktopNotesList(page)
    const cdp = await page.context().newCDPSession(page)
    await cdp.send("Emulation.setPageScaleFactor", { pageScaleFactor: 2 })
    await expect
      .poll(() => page.evaluate(() => window.visualViewport?.scale ?? 1))
      .toBeCloseTo(2, 1)
    visualEvidence.pageScale200 = {
      scale: await page.evaluate(() => window.visualViewport?.scale ?? 1),
      origin: await assertPageScaleOriginContract(page)
    }
    visualEvidence.pageScale200Screenshot = await screenshot(page, "page-scale-200")
    await cdp.send("Emulation.setPageScaleFactor", { pageScaleFactor: 1 })

    await page.setViewportSize({ width: 720, height: 900 })
    await closeMobileNotesList(page)
    visualEvidence.effectiveReflow720 = await assertVisualContract(page)
    visualEvidence.effectiveReflow720InspectorViewport = await bringResponsiveInspectorIntoView(page)
    visualEvidence.effectiveReflow720Screenshot = await screenshot(
      page,
      "effective-reflow-720"
    )

    await openSuggestions(page)
    const graphCallsBeforeAccept = fixture.graphCalls.length
    await page.getByRole("button", { name: `Accept ${LONG_TARGET}` }).click()
    await expect(
      page.locator('[data-suggestion-review-row="related-suggestion"]')
    ).toHaveCount(0)
    await expect.poll(() => fixture.graphCalls.length).toBeGreaterThan(graphCallsBeforeAccept)

    await page.getByRole("button", { name: "Relationships", exact: true }).click()
    const relationships = page.getByTestId("notes-graph-relationships-view")
    await expect(relationships).toBeVisible()
    await expect(relationships.getByTestId("notes-graph-relationship-row")).toHaveCount(100)
    await relationships.getByRole("button", { name: "Next page" }).click()
    await expect(relationships.getByText("2 / 2", { exact: true })).toBeVisible()
    await expect(relationships.getByTestId("notes-graph-relationship-row").first()).toBeFocused()

    await page.getByRole("button", { name: "Canvas", exact: true }).click()
    await openSuggestions(page)
    await page.getByRole("button", { name: `Reject ${LONG_TAG}` }).click()
    await expect(page.getByText(LONG_TAG, { exact: true })).toHaveCount(0)
    await page.getByRole("button", { name: "Suggestion actions" }).click()
    await page.getByRole("menuitem", { name: "Reset dismissed suggestions" }).click()
    const dialog = page.getByRole("dialog", { name: "Reset dismissed suggestions?" })
    await expect(dialog).toBeVisible()
    await dialog.getByRole("button", { name: "Reset dismissed" }).click()
    await expect(page.getByText(LONG_TAG, { exact: true })).toBeVisible()

    await page.context().setOffline(true)
    await page.evaluate(() => {
      const store = (
        window as typeof window & {
          __tldw_useConnectionStore: {
            getState: () => { state: Record<string, unknown> }
            setState: (state: { state: Record<string, unknown> }) => void
          }
        }
      ).__tldw_useConnectionStore
      const current = store.getState().state
      store.setState({
        state: {
          ...current,
          phase: "error",
          isConnected: false,
          isChecking: false,
          errorKind: "unreachable",
          consecutiveFailures: 3,
          lastError: "offline",
          lastStatusCode: 0,
          lastCheckedAt: Date.now()
        }
      })
    })
    await expect(page.getByTestId("notes-graph-offline-state")).toBeVisible()
    await expect(page.getByRole("button", { name: `Accept ${LONG_TAG}` })).toBeDisabled()
    expect((await canvasPixels(page)).paintedPixels).toBeGreaterThan(50)
    await page.context().setOffline(false)

    assertExactSuggestionRequests(fixture.calls, {
      [`GET ${SUGGESTION_BASE}/capabilities`]: 2,
      [`GET ${SUGGESTION_BASE}/runs`]: 3,
      [`GET ${SUGGESTION_BASE}/runs/run-1`]: 4,
      [`GET ${SUGGESTION_BASE}`]: 4,
      [`POST ${SUGGESTION_BASE}/related-suggestion/accept`]: 1,
      [`POST ${SUGGESTION_BASE}/rejections/reset`]: 1,
      [`POST ${SUGGESTION_BASE}/runs`]: 1,
      [`POST ${SUGGESTION_BASE}/tag-suggestion/reject`]: 1
    })
    await attachEvidence(testInfo, fixture, visualEvidence)
  })

  test("surfaces cancellation, stale, and failure terminal states", async ({
    authedPage: page
  }) => {
    const fixture = new NotesGraphFixture({
      outcomes: ["succeeded", "stale", "failed"]
    })
    await installFixture(page, fixture)
    await openGraph(page)
    await openSuggestions(page)

    await page.getByRole("button", { name: "Generate", exact: true }).click()
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Queued"
    )
    await expect.poll(() => fixture.count("GET", "/runs/run-1")).toBeGreaterThanOrEqual(1)
    await page.getByRole("button", { name: "Cancel generation" }).click()
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Cancelled",
      { timeout: 10_000 }
    )

    await page.getByRole("button", { name: "Regenerate" }).click()
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Queued"
    )
    await expect.poll(() => fixture.count("GET", "/runs/run-2")).toBeGreaterThanOrEqual(1)
    await advanceRun(page, fixture, "run-2", "stale")

    await page.getByRole("button", { name: "Regenerate" }).click()
    await expect(page.getByTestId("notes-graph-suggestion-run-status")).toHaveText(
      "Queued"
    )
    await expect.poll(() => fixture.count("GET", "/runs/run-3")).toBeGreaterThanOrEqual(1)
    await advanceRun(page, fixture, "run-3", "failed")
    expect(fixture.admissionCalls).toHaveLength(3)
    assertExactSuggestionRequests(fixture.calls, {
      [`GET ${SUGGESTION_BASE}/capabilities`]: 1,
      [`GET ${SUGGESTION_BASE}/runs`]: 4,
      [`GET ${SUGGESTION_BASE}/runs/run-1`]: 2,
      [`GET ${SUGGESTION_BASE}/runs/run-2`]: 4,
      [`GET ${SUGGESTION_BASE}/runs/run-3`]: 4,
      [`GET ${SUGGESTION_BASE}`]: 1,
      [`POST ${SUGGESTION_BASE}/runs`]: 3,
      [`POST ${SUGGESTION_BASE}/runs/run-1/cancel`]: 1
    })
  })

  test("keeps read-only and non-note scopes free of nested suggestion requests", async ({
    authedPage: page
  }) => {
    const readOnly = new NotesGraphFixture({ authorized: false })
    await installFixture(page, readOnly)
    await openGraph(page)
    await expect(page.getByRole("tab", { name: "Suggestions" })).toHaveCount(0)
    await expect(page.getByRole("button", { name: "Generate" })).toHaveCount(0)
    await page.waitForTimeout(500)
    assertExactSuggestionRequests(readOnly.calls, {})

    await page.unroute("**/api/v1/**")
    const authorized = new NotesGraphFixture()
    await installFixture(page, authorized)
    await page.reload({ waitUntil: "domcontentloaded" })
    const graphEntry = page.getByTestId("notes-view-mode-graph")
    await expect(graphEntry).toBeVisible()
    await graphEntry.click()
    await expect(page.getByTestId("notes-graph-canvas")).toBeVisible()
    await expect(page.getByRole("tab", { name: "Suggestions" })).toBeVisible()
    await expect.poll(() => authorized.calls.length).toBe(3)
    assertExactSuggestionRequests(authorized.calls, {
      [`GET ${SUGGESTION_BASE}/capabilities`]: 1,
      [`GET ${SUGGESTION_BASE}/runs`]: 1,
      [`GET ${SUGGESTION_BASE}`]: 1
    })
    const beforeNonNote = authorized.calls.length
    await page
      .getByRole("button", { name: "Web source with a long canonical label" })
      .first()
      .click()
    await expect(page.getByRole("tab", { name: "Suggestions" })).toHaveCount(0)
    await page.waitForTimeout(750)
    expect(authorized.calls).toHaveLength(beforeNonNote)
    assertExactSuggestionRequests(authorized.calls, {
      [`GET ${SUGGESTION_BASE}/capabilities`]: 1,
      [`GET ${SUGGESTION_BASE}/runs`]: 1,
      [`GET ${SUGGESTION_BASE}`]: 1
    })
  })
})
