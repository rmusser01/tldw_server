import { expect, test, type Page } from "@playwright/test"

import { seedAuth } from "../utils/helpers"
import {
  ResearchWorkspacePage,
  classifySharedRecipientForbiddenRequest,
  startSharedRecipientRequestLedger
} from "../utils/page-objects/ResearchWorkspacePage"

const SHARE_ID = 42
const API_ROOT = `/api/v1/sharing/shared-with-me/${SHARE_ID}`
const GENERATED_AT = "2026-08-22T17:00:00Z"

type StubState = {
  revoked: boolean
  messages: Array<Record<string, unknown>>
  chatRequests: Array<Record<string, unknown>>
}

const source = (
  sourceId: string,
  title: string,
  state = "ready"
): Record<string, unknown> => ({
  source_id: sourceId,
  title,
  source_type: "pdf",
  origin_url: "https://evidence.example.test",
  origin_host: "evidence.example.test",
  state,
  reason_code: state === "ready" ? null : `source_${state}`,
  citation_ready: state === "ready",
  retrieval_ready: state === "ready",
  position: sourceId === "source-1" ? 0 : 1,
  added_at: GENERATED_AT
})

const SOURCES = [
  source("source-1", "First shared source"),
  source("source-2", "Second shared source")
]

const pagination = (items: Array<Record<string, unknown>>) => ({
  offset: 0,
  limit: 50,
  total: items.length,
  has_more: false
})

const summary = (items: Array<Record<string, unknown>>) => ({
  total: items.length,
  queryable: items.filter((item) => item.retrieval_ready).length,
  processing: 0,
  failed: 0
})

const initialMessages: Array<Record<string, unknown>> = [
  {
    message_id: "history-user-1",
    role: "user",
    content: "What is already known?",
    created_at: "2026-08-22T16:55:00Z",
    citations: []
  },
  {
    message_id: "history-assistant-1",
    role: "assistant",
    content: "The first source establishes the baseline.",
    created_at: "2026-08-22T16:55:01Z",
    citations: [
      {
        citation_id: "history-citation-1",
        source_id: "source-1",
        source_title: "First shared source",
        locator: { chunk: 1, start_char: 0, end_char: 22 },
        quote: "Baseline evidence text.",
        score: 0.91
      }
    ]
  }
]

const bootstrap = (state: StubState) => ({
  schema_version: 1,
  generated_at: GENERATED_AT,
  share: {
    share_id: SHARE_ID,
    access_level: "view_chat",
    allow_clone: false,
    owner_display_name: "Research owner",
    shared_at: "2026-08-21T20:00:00Z"
  },
  workspace: {
    workspace_id: "owner-workspace-42",
    name: "Recipient evidence review",
    description: "Canonical owner source set"
  },
  allowed_actions: {
    inspect_sources: { allowed: true, reason_code: null },
    ask_grounded_questions: { allowed: true, reason_code: null },
    add_sources: { allowed: false, reason_code: "shared_write_not_available" },
    edit_workspace: {
      allowed: false,
      reason_code: "shared_write_not_available"
    },
    clone_workspace: { allowed: false, reason_code: "clone_deferred" }
  },
  generation_default: {
    provider: "openai",
    model: "shared-test-model",
    ready: true,
    reason_code: null
  },
  source_summary: summary(SOURCES),
  sources: { items: SOURCES, pagination: pagination(SOURCES) },
  conversation: {
    conversation_id: state.messages.length ? "shared-conversation-42" : null,
    messages: state.messages,
    next_before: null
  },
  partial_errors: []
})

const installSharedApi = async (page: Page, state: StubState): Promise<void> => {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method().toUpperCase()

    if (path === "/api/v1/llm/models/metadata") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            {
              id: "shared-test-model",
              name: "Shared test model",
              provider: "openai",
              capabilities: ["chat"]
            }
          ]
        })
      })
      return
    }

    if (state.revoked && path.startsWith(API_ROOT)) {
      await route.fulfill({
        status: 404,
        contentType: "application/json",
        body: JSON.stringify({
          detail: {
            code: "shared_workspace_not_found",
            message: "Shared workspace not found.",
            retryable: false
          }
        })
      })
      return
    }

    if (path === `${API_ROOT}/workspace` && method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(bootstrap(state))
      })
      return
    }

    if (path === `${API_ROOT}/sources` && method === "GET") {
      const query = (url.searchParams.get("q") || "").toLowerCase()
      const requestedState = url.searchParams.get("state")
      const items = SOURCES.filter(
        (item) =>
          (!query || String(item.title).toLowerCase().includes(query)) &&
          (!requestedState || item.state === requestedState)
      )
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          items,
          pagination: pagination(items),
          summary: summary(SOURCES),
          partial_errors: []
        })
      })
      return
    }

    const previewMatch = path.match(
      new RegExp(`^${API_ROOT}/sources/(source-[12])/preview$`)
    )
    if (previewMatch && method === "GET") {
      const sourceId = previewMatch[1]
      const title = sourceId === "source-1" ? "First shared source" : "Second shared source"
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          source_id: sourceId,
          title,
          source_type: "pdf",
          origin_url: "https://evidence.example.test",
          origin_host: "evidence.example.test",
          state: "ready",
          reason_code: null,
          content_available: true,
          preview_mode: "content_excerpt",
          unavailable_reason: null,
          text_preview: `Preview evidence for ${title}.`,
          text_total_chars: 128,
          text_truncated: false,
          snippets: [],
          generated_at: GENERATED_AT
        })
      })
      return
    }

    if (path === `${API_ROOT}/chat/messages` && method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          conversation_id: "shared-conversation-42",
          messages: state.messages,
          next_before: null
        })
      })
      return
    }

    if (path === `${API_ROOT}/chat` && method === "POST") {
      const payload = request.postDataJSON() as Record<string, unknown>
      const scope = payload.source_scope as {
        mode: "all" | "include"
        source_ids: string[]
      }
      state.chatRequests.push(payload)
      const userMessage = {
        message_id: "submitted-user-1",
        role: "user",
        content: payload.query,
        created_at: "2026-08-22T17:01:00Z",
        citations: []
      }
      const assistantMessage = {
        message_id: "submitted-assistant-1",
        role: "assistant",
        content: "The second source contains the selected evidence.",
        created_at: "2026-08-22T17:01:01Z",
        citations: [
          {
            citation_id: "submitted-citation-1",
            source_id: "source-2",
            source_title: "Second shared source",
            locator: { chunk: 2, start_char: 0, end_char: 24 },
            quote: "Second source evidence.",
            score: 0.94
          }
        ]
      }
      state.messages.push(userMessage, assistantMessage)
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          schema_version: 1,
          request_id: payload.request_id,
          conversation_id: "shared-conversation-42",
          turn: { user_message: userMessage, assistant_message: assistantMessage },
          citations: assistantMessage.citations,
          generation: { provider: "openai", model: "shared-test-model" },
          source_scope: {
            mode: scope.mode,
            effective_source_count:
              scope.mode === "all" ? SOURCES.length : scope.source_ids.length
          },
          replay: { replayed: false }
        })
      })
      return
    }

    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Unexpected API request in shared stub" })
    })
  })
}

test.beforeEach(async ({ page }) => {
  await seedAuth(page)
})

test("desktop recipient flow uses only canonical shared reads and chat", async ({
  page
}) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  const state: StubState = {
    revoked: false,
    messages: [...initialMessages],
    chatRequests: []
  }
  await installSharedApi(page, state)
  const ledger = startSharedRecipientRequestLedger(page, SHARE_ID)
  const workspace = new ResearchWorkspacePage(page)

  await workspace.gotoShared(SHARE_ID)
  await workspace.waitForSharedReady("Recipient evidence review")
  await expect(page.getByText("First shared source", { exact: true })).toBeVisible()
  await expect(page.getByText("Second shared source", { exact: true })).toBeVisible()

  await workspace.searchSharedSources("First")
  await expect(page.getByText("Second shared source", { exact: true })).toBeHidden()
  await workspace.searchSharedSources("")
  await workspace.filterSharedSourcesByState("ready")
  await workspace.clearSharedSourceSelection()
  await workspace.selectSharedSource("Second shared source")
  await workspace.previewSharedSource("Second shared source")
  await expect(page.getByRole("dialog", { name: "Source preview" })).toContainText(
    "Preview evidence for Second shared source."
  )
  await workspace.closeSharedSourcePreview()

  await workspace.askSharedWorkspace("What does the selected source show?")
  await expect(page.getByText("The second source contains the selected evidence.")).toBeVisible()
  expect(state.chatRequests).toHaveLength(1)
  expect(state.chatRequests[0]?.source_scope).toEqual({
    mode: "include",
    source_ids: ["source-2"]
  })

  await workspace.openSharedCitation("Second shared source")
  await expect(page.getByRole("dialog", { name: "Source preview" })).toContainText(
    "Second shared source"
  )
  await workspace.closeSharedSourcePreview()

  await page.reload({ waitUntil: "domcontentloaded" })
  await workspace.waitForSharedReady("Recipient evidence review")
  await expect(page.getByText("The second source contains the selected evidence.")).toBeVisible()
  ledger.assertClean()
  const canonicalOperations = ledger
    .snapshot()
    .filter((entry) => entry.pathname.startsWith(API_ROOT))
    .map((entry) => `${entry.method} ${entry.pathname}`)
  expect(canonicalOperations).toEqual(
    expect.arrayContaining([
      `GET ${API_ROOT}/workspace`,
      `GET ${API_ROOT}/sources`,
      `GET ${API_ROOT}/sources/source-2/preview`,
      `POST ${API_ROOT}/chat`
    ])
  )
  ledger.dispose()
})

test("mobile tabs remain scoped and revoked reload fails closed", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  const state: StubState = {
    revoked: false,
    messages: [...initialMessages],
    chatRequests: []
  }
  await installSharedApi(page, state)
  const ledger = startSharedRecipientRequestLedger(page, SHARE_ID)
  const workspace = new ResearchWorkspacePage(page)

  await workspace.gotoShared(SHARE_ID)
  await workspace.waitForSharedReady("Recipient evidence review")
  await workspace.activateSharedMobilePane("chat")
  await expect(page.getByTestId("shared-workspace-chat-pane")).toBeVisible()
  await workspace.activateSharedMobilePane("sources")
  await expect(page.getByTestId("shared-workspace-sources-pane")).toBeVisible()

  state.revoked = true
  await page.reload({ waitUntil: "domcontentloaded" })
  await expect(page.getByText("This shared workspace isn't available.")).toBeVisible()
  await expect(page.getByRole("link", { name: "Return to Shared with me" })).toBeVisible()
  ledger.assertClean()
  ledger.dispose()
})

test("request ledger classifies every prohibited shared-mode destination", () => {
  const probes: Array<[string, string]> = [
    ["GET", "/api/v1/workspaces/local-workspace/context"],
    ["PATCH", "/api/v1/sharing/workspaces/owner-workspace"],
    ["POST", "/api/v1/studio/generate"],
    ["GET", "/api/v1/notes"],
    ["POST", "/api/v1/mcp/tools/execute"],
    ["POST", "/api/v1/acp/sessions"],
    ["POST", "/api/v1/sandbox/run"],
    ["GET", "/api/v1/artifacts/123"],
    ["POST", `${API_ROOT}/sources`],
    ["POST", "/api/v1/media/process-documents"],
    ["GET", `${API_ROOT}/media/99`]
  ]

  for (const [method, pathname] of probes) {
    expect(
      classifySharedRecipientForbiddenRequest(
        `http://127.0.0.1:8000${pathname}`,
        method,
        SHARE_ID
      ),
      `${method} ${pathname}`
    ).not.toBeNull()
  }
})
