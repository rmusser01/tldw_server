import { type Page, type Route } from "@playwright/test"
import { test, expect, assertNoCriticalErrors } from "./utils/fixtures"
import { stubNotificationsApi } from "./utils/helpers"
import { ExplainerPage } from "./utils/page-objects"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }

type MockExplainerNode = {
  id: string
  sessionId: string
  parentId: string | null
  ordinal: number
  title: string
  body: string | null
  kind: string
  intent: string
  status: string
  evidenceState: string
  outsideKnowledgeUsed: boolean
  citations: Array<Record<string, unknown>>
  questionOptions: Array<Record<string, unknown>> | null
  selectedOptionId: string | null
  selectedCustomAnswer: string | null
  generationMetadata: Record<string, unknown> | null
  childNodeIds: string[]
  createdAt: string
  updatedAt: string
}

type MockExplainerSession = {
  id: string
  ownerUserId: string
  title: string
  mode: string
  status: string
  outputIntent: string
  grounding: string
  depthPreset: string
  selectedSources: Array<Record<string, unknown>>
  rootNodeIds: string[]
  nodes: Record<string, MockExplainerNode>
  createdAt: string
  updatedAt: string
  archivedAt: string | null
}

type MockExplainerJob = {
  jobId: string
  sessionId: string
  nodeId: string
  status: string
}

const json = async (route: Route, status: number, payload: unknown): Promise<void> => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload)
  })
}

const now = () => "2026-06-09T00:00:00Z"

const makeNode = (
  sessionId: string,
  overrides: Partial<MockExplainerNode> & { id: string; title: string }
): MockExplainerNode => ({
  id: overrides.id,
  sessionId,
  parentId: null,
  ordinal: 0,
  title: overrides.title,
  body: null,
  kind: "explanation",
  intent: "explain",
  status: "complete",
  evidenceState: "supported",
  outsideKnowledgeUsed: false,
  citations: [],
  questionOptions: null,
  selectedOptionId: null,
  selectedCustomAnswer: null,
  generationMetadata: null,
  childNodeIds: [],
  createdAt: now(),
  updatedAt: now(),
  ...overrides
})

const summarizeSession = (session: MockExplainerSession) => ({
  id: session.id,
  ownerUserId: session.ownerUserId,
  title: session.title,
  mode: session.mode,
  status: session.status,
  outputIntent: session.outputIntent,
  grounding: session.grounding,
  depthPreset: session.depthPreset,
  nodeCount: Object.keys(session.nodes).length,
  selectedSourceCount: session.selectedSources.length,
  createdAt: session.createdAt,
  updatedAt: session.updatedAt,
  archivedAt: session.archivedAt
})

const installExplainerMocks = async (page: Page) => {
  const sessions = new Map<string, MockExplainerSession>()
  const jobs = new Map<string, MockExplainerJob>()
  let nextSession = 1
  let nextJob = 1

  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    await json(route, 200, { status: "ok", version: "e2e" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
    await json(route, 200, { openapi: "3.0.0", paths: {} })
  })

  await page.route(/\/api\/v1\/config\/docs-info(?:\?.*)?$/, async (route) => {
    await json(route, 200, { available: true })
  })

  await stubNotificationsApi(page)

  await page.route(/\/api\/v1\/media\/search(?:\?.*)?$/, async (route) => {
    await json(route, 200, {
      items: [
        {
          media_id: "media-42",
          title: "Attention Source PDF",
          media_type: "pdf",
          url: "https://example.test/attention.pdf"
        }
      ],
      total: 1
    })
  })

  await page.route(/\/api\/v1\/notes\/search\/(?:\?.*)?$/, async (route) => {
    await json(route, 200, {
      notes: [
        {
          id: "note-7",
          title: "Attention note",
          content: "Query-key similarity note"
        }
      ],
      total: 1
    })
  })

  await page.route(/\/api\/v1\/explainer(?:\/.*)?(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const method = request.method().toUpperCase()
    const { pathname } = new URL(request.url())

    if (method === "GET" && pathname === "/api/v1/explainer/sessions") {
      const items = Array.from(sessions.values()).map(summarizeSession)
      await json(route, 200, {
        items,
        total: items.length,
        limit: 50,
        offset: 0
      })
      return
    }

    if (method === "POST" && pathname === "/api/v1/explainer/sessions") {
      const payload = request.postDataJSON() as {
        title: string
        mode: "goal" | "sources"
        outputIntent: string
        grounding: string
        depthPreset: string
        rootPrompt: string
        selectedSources?: Array<Record<string, unknown>>
      }
      const id = `session-${nextSession++}`
      const rootId = `root-${id}`
      const isSourceSession = payload.mode === "sources"
      const root = makeNode(id, {
        id: rootId,
        title: payload.rootPrompt,
        body: isSourceSession
          ? "Source-led explanation from Attention Source PDF."
          : "Attention lets tokens route information to each other.",
        intent: payload.outputIntent,
        citations: isSourceSession
          ? [
              {
                id: "cite-1",
                sourceId: "media-42",
                sourceType: "media",
                title: "Attention Source PDF",
                excerpt: "Attention weights are computed from query-key similarity.",
                locationLabel: "chunk 3",
                snapshotHash: "sha256:e2e"
              }
            ]
          : []
      })
      const session: MockExplainerSession = {
        id,
        ownerUserId: "7",
        title: payload.title,
        mode: payload.mode,
        status: "active",
        outputIntent: payload.outputIntent,
        grounding: payload.grounding,
        depthPreset: payload.depthPreset,
        selectedSources: payload.selectedSources ?? [],
        rootNodeIds: [rootId],
        nodes: { [rootId]: root },
        createdAt: now(),
        updatedAt: now(),
        archivedAt: null
      }
      sessions.set(id, session)
      await json(route, 200, session)
      return
    }

    const sessionMatch = pathname.match(/^\/api\/v1\/explainer\/sessions\/([^/]+)$/)
    if (method === "GET" && sessionMatch) {
      const session = sessions.get(sessionMatch[1])
      await json(route, session ? 200 : 404, session ?? { detail: "Session not found" })
      return
    }

    const expandMatch = pathname.match(
      /^\/api\/v1\/explainer\/sessions\/([^/]+)\/nodes\/([^/]+)\/expand$/
    )
    if (method === "POST" && expandMatch) {
      const [, sessionId, nodeId] = expandMatch
      const session = sessions.get(sessionId)
      const node = session?.nodes[nodeId]
      if (!session || !node) {
        await json(route, 404, { detail: "Node not found" })
        return
      }
      node.status = "generating"
      const jobId = `job-${nextJob++}`
      jobs.set(jobId, { jobId, sessionId, nodeId, status: "queued" })
      await json(route, 202, { jobId, sessionId, nodeId, status: "queued" })
      return
    }

    const jobMatch = pathname.match(/^\/api\/v1\/explainer\/jobs\/([^/]+)$/)
    if (method === "GET" && jobMatch) {
      const job = jobs.get(jobMatch[1])
      const session = job ? sessions.get(job.sessionId) : null
      const node = job && session ? session.nodes[job.nodeId] : null
      if (!job || !session || !node) {
        await json(route, 404, { detail: "Job not found" })
        return
      }
      if (job.status !== "completed") {
        const childId = `child-${job.nodeId}`
        node.status = "complete"
        node.childNodeIds = [childId]
        session.nodes[childId] = makeNode(session.id, {
          id: childId,
          parentId: node.id,
          ordinal: 1,
          title: "Scaled dot-product attention",
          body: "Scaled dot-product attention compares query and key vectors, then mixes values."
        })
        session.updatedAt = now()
        job.status = "completed"
      }
      await json(route, 200, {
        jobId: job.jobId,
        sessionId: job.sessionId,
        nodeId: job.nodeId,
        status: "completed",
        progressPercent: 100,
        progressMessage: "Expansion complete"
      })
      return
    }

    const exportMatch = pathname.match(
      /^\/api\/v1\/explainer\/sessions\/([^/]+)\/export-chatbook$/
    )
    if (method === "POST" && exportMatch) {
      await json(route, 200, {
        success: true,
        message: "Export job started: chatbook-job-1",
        job_id: "chatbook-job-1"
      })
      return
    }

    await json(route, 404, { detail: `Unhandled Explainer mock route: ${method} ${pathname}` })
  })
}

test.describe("Explainer workflow", () => {
  test.beforeEach(async ({ authedPage }) => {
    await authedPage.setViewportSize(DESKTOP_VIEWPORT)
    await installExplainerMocks(authedPage)
  })

  test("loads the Explainer workbench shell", async ({ authedPage, diagnostics }) => {
    const explainer = new ExplainerPage(authedPage)

    await explainer.goto()

    await expect(explainer.heading).toBeVisible()
    await expect(explainer.goalTab).toBeVisible()
    await expect(explainer.sourcesTab).toBeVisible()
    await expect(authedPage.getByRole("button", { name: "Export to Chatbook" })).toBeDisabled()
    await assertNoCriticalErrors(diagnostics)
  })

  test("creates a goal session, completes expansion polling, and exports to Chatbook", async ({
    authedPage,
    diagnostics
  }) => {
    const explainer = new ExplainerPage(authedPage)

    await explainer.goto()
    await explainer.createGoalSession("Explain transformer attention")

    await expect(authedPage.getByText("Attention lets tokens route information to each other."))
      .toBeVisible()
    await explainer.expectNodeStatus(/Complete/)

    await explainer.expandSelectedNode()
    await expect(authedPage.getByText("Scaled dot-product attention")).toBeVisible()
    await explainer.selectNode("Scaled dot-product attention")
    await expect(
      authedPage.getByText("Scaled dot-product attention compares query and key vectors")
    ).toBeVisible()

    await explainer.exportToChatbook()
    await expect(authedPage.getByText("Export job started: chatbook-job-1")).toBeVisible()
    await assertNoCriticalErrors(diagnostics)
  })

  test("creates a source-grounded session from the page-owned source picker", async ({
    authedPage,
    diagnostics
  }) => {
    const explainer = new ExplainerPage(authedPage)

    await explainer.goto()
    await explainer.openSourcesTab()
    await expect(authedPage.getByLabel("Grounding mode")).toHaveValue("source_led")
    await authedPage.getByLabel("Grounding mode").selectOption("source_only")
    await authedPage.getByLabel("Output intent").selectOption("both")
    await explainer.searchSource("attention")

    await expect(authedPage.getByText("Attention Source PDF")).toBeVisible()
    await explainer.selectFirstSource()
    await explainer.createSourceSession()

    await expect(authedPage.getByText("Source-led explanation from Attention Source PDF."))
      .toBeVisible()
    await explainer.expectCitation(/Attention weights are computed from query-key similarity/)
    await assertNoCriticalErrors(diagnostics)
  })
})
