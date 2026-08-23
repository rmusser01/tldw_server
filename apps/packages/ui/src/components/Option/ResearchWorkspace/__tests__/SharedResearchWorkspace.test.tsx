import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { I18nextProvider } from "react-i18next"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { TldwApiError } from "@/services/tldw/api-error"
import { SharedResearchWorkspace } from "../SharedResearchWorkspace"
import {
  buildBootstrap,
  chatResponse,
  createSharedWorkspaceTestI18n,
  preview,
  sourcePage
} from "./shared-research-workspace-test-utils"

let testI18n: Awaited<ReturnType<typeof createSharedWorkspaceTestI18n>>

const { api, fetchChatModels } = vi.hoisted(() => ({
  api: {
    bootstrap: vi.fn(),
    listSources: vi.fn(),
    previewSource: vi.fn(),
    listMessages: vi.fn(),
    ask: vi.fn()
  },
  fetchChatModels: vi.fn()
}))

vi.mock(
  "@/services/tldw/domains/shared-workspaces",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/services/tldw/domains/shared-workspaces")
      >()
    return { ...actual, sharedWorkspacesApi: api }
  }
)

vi.mock("@/services/tldw-server", () => ({ fetchChatModels }))

const renderWorkspace = () =>
  render(
    <MemoryRouter>
      <I18nextProvider i18n={testI18n}>
        <SharedResearchWorkspace shareId={42} />
      </I18nextProvider>
    </MemoryRouter>
  )

describe("SharedResearchWorkspace recipient surface", () => {
  beforeAll(async () => {
    testI18n = await createSharedWorkspaceTestI18n()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.bootstrap.mockResolvedValue(buildBootstrap())
    api.listSources.mockResolvedValue(sourcePage)
    api.previewSource.mockResolvedValue(preview)
    api.listMessages.mockResolvedValue({
      conversation_id: "conversation-1",
      messages: [
        {
          message_id: "message-older",
          role: "user",
          content: "Older question",
          created_at: "2026-08-20T10:00:00Z",
          citations: []
        },
        {
          message_id: "message-existing",
          role: "assistant",
          content: "Existing **grounded** answer.",
          created_at: "2026-08-21T11:30:00Z",
          citations: []
        }
      ],
      next_before: null
    })
    api.ask.mockImplementation((_shareId, request) =>
      Promise.resolve({ ...chatResponse, request_id: request.request_id })
    )
    fetchChatModels.mockResolvedValue([
      { model: "generic-model", provider: "openai", configured: true }
    ])
  })

  it("renders identity, policy ceiling, sources, and only server-authorized controls", async () => {
    renderWorkspace()

    expect(
      await screen.findByRole("heading", { name: "Election evidence review" })
    ).toBeInTheDocument()
    expect(screen.getByText("Shared by Avery Owner")).toBeInTheDocument()
    expect(
      screen.getByLabelText("Shared workspace capabilities")
    ).toHaveTextContent("Can ask questions")
    const tier = screen.getByLabelText("Access tier: view_chat_add")
    expect(tier).toHaveTextContent("view_chat_add")
    fireEvent.mouseOver(tier)
    expect(
      await screen.findByText(
        "This access level is the owner's policy ceiling. Editing shared content is not available here yet."
      )
    ).toBeInTheDocument()

    expect(screen.queryByText("Studio")).not.toBeInTheDocument()
    expect(screen.queryByText("General Chat")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /add source/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /edit workspace/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /clone/i })).not.toBeInTheDocument()

    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeChecked()
    expect(
      screen.getByRole("checkbox", { name: "Select Processing interview" })
    ).toBeDisabled()
    expect(screen.getByText("Transcription pending")).toBeInTheDocument()
    expect(screen.getByText("1 of 1 queryable sources selected")).toBeInTheDocument()
  })

  it("keeps bulk scope explicit and sends no question with an empty subset", async () => {
    renderWorkspace()
    await screen.findByText("Queryable report")

    fireEvent.click(screen.getByRole("button", { name: "Clear selected sources" }))
    expect(screen.getByText("0 of 1 queryable sources selected")).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "Should not submit" }
    })
    expect(screen.getByRole("button", { name: "Ask shared workspace" })).toBeDisabled()
    expect(api.ask).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Select all queryable sources" }))
    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeChecked()
  })

  it("requires an explicit subset when all scope exceeds 500 sources", async () => {
    const bootstrap = buildBootstrap()
    api.bootstrap.mockResolvedValue({
      ...bootstrap,
      source_summary: {
        total: 501,
        queryable: 501,
        processing: 0,
        failed: 0
      }
    })
    renderWorkspace()

    expect(
      await screen.findByText(
        "Clear the selection, then choose up to 500 sources."
      )
    ).toBeInTheDocument()
    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Clear selected sources" })).toBeEnabled()
    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "Summarize the selected source" }
    })
    expect(screen.getByRole("button", { name: "Ask shared workspace" })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Clear selected sources" }))
    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeEnabled()
    fireEvent.click(screen.getByRole("checkbox", { name: "Select Queryable report" }))
    expect(
      screen.queryByText(/Clear the selection, then choose up to 500 sources/)
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Ask shared workspace" })).toBeEnabled()
  })

  it("shows a compact fail-closed state while all-mode selection is materialized", async () => {
    const readySources = Array.from({ length: 50 }, (_, index) => ({
      ...sourcePage.items[0],
      source_id: `source-${index + 1}`,
      title: index === 0 ? "Queryable report" : `Source ${index + 1}`,
      position: index + 1
    }))
    api.bootstrap.mockResolvedValue(
      buildBootstrap({
        source_summary: {
          total: 75,
          queryable: 75,
          processing: 0,
          failed: 0
        },
        sources: {
          items: readySources,
          pagination: {
            offset: 0,
            limit: 50,
            total: 75,
            has_more: true
          }
        }
      })
    )
    let rejectSources: (reason?: unknown) => void = () => undefined
    api.listSources.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectSources = reject
        })
    )
    renderWorkspace()
    const checkbox = await screen.findByRole("checkbox", {
      name: "Select Queryable report"
    })

    fireEvent.click(checkbox)

    expect(
      await screen.findByText("Preparing complete source selection...")
    ).toBeInTheDocument()
    expect(checkbox).toBeChecked()
    expect(checkbox).toBeDisabled()

    rejectSources(new TypeError("connection reset"))

    expect(
      await screen.findByText(
        "Couldn't load every queryable source. All queryable sources remain selected."
      )
    ).toBeInTheDocument()
    expect(checkbox).toBeChecked()
  })

  it("keeps a literal view tier separate from an allowed chat capability", async () => {
    const bootstrap = buildBootstrap()
    api.bootstrap.mockResolvedValue({
      ...bootstrap,
      share: { ...bootstrap.share, access_level: "view" }
    })

    renderWorkspace()

    expect(
      await screen.findByLabelText("Shared workspace capabilities")
    ).toHaveTextContent("Can ask questions")
    expect(screen.getByLabelText("Access tier: view")).toHaveTextContent("view")
  })

  it("treats server allowed_actions and its reason as the chat authority", async () => {
    const bootstrap = buildBootstrap()
    api.bootstrap.mockResolvedValue({
      ...bootstrap,
      share: { ...bootstrap.share, access_level: "full_edit" },
      allowed_actions: {
        ...bootstrap.allowed_actions,
        ask_grounded_questions: {
          allowed: false,
          reason_code: "share_read_only"
        }
      }
    })
    renderWorkspace()
    expect(
      await screen.findByLabelText("Shared workspace capabilities")
    ).toHaveTextContent("View only: share read only")
    expect(screen.getByLabelText("Access tier: full_edit")).toHaveTextContent(
      "full_edit"
    )

    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "This must remain local" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))
    expect(screen.getByRole("button", { name: "Ask shared workspace" })).toBeDisabled()
    expect(api.ask).not.toHaveBeenCalled()
  })

  it("uses inspect_sources as the sole authority for every source control", async () => {
    const bootstrap = buildBootstrap()
    api.bootstrap.mockResolvedValue({
      ...bootstrap,
      share: { ...bootstrap.share, access_level: "full_edit" },
      allowed_actions: {
        ...bootstrap.allowed_actions,
        inspect_sources: {
          allowed: false,
          reason_code: "workspace_inspection_disabled"
        },
        ask_grounded_questions: {
          allowed: false,
          reason_code: "no_provider_configured"
        }
      },
      sources: {
        ...bootstrap.sources,
        pagination: {
          ...bootstrap.sources.pagination,
          total: 52,
          has_more: true
        }
      }
    })
    renderWorkspace()

    expect(
      await screen.findByLabelText("Shared workspace capabilities")
    ).toHaveTextContent(
      "Access restricted: workspace inspection disabled"
    )
    expect(screen.queryByText("Can ask questions")).not.toBeInTheDocument()
    expect(screen.getByText("workspace inspection disabled")).toBeInTheDocument()
    expect(screen.getByText("no provider configured")).toBeInTheDocument()

    expect(screen.getByLabelText("Search shared sources")).toBeDisabled()
    expect(screen.getByLabelText("Filter shared sources by state")).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Select all queryable sources" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Clear selected sources" })
    ).toBeDisabled()
    expect(
      screen.getByRole("checkbox", { name: "Select Queryable report" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Preview Queryable report" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Next source page" })
    ).toBeDisabled()

    fireEvent.click(
      screen.getByRole("button", { name: "Preview Queryable report" })
    )
    expect(api.previewSource).not.toHaveBeenCalled()
    expect(api.listSources).not.toHaveBeenCalled()
  })

  it("uses server search, state filters, and pagination without dropping scope", async () => {
    api.listSources.mockResolvedValue({
      ...sourcePage,
      pagination: { offset: 50, limit: 50, total: 101, has_more: true }
    })
    renderWorkspace()
    await screen.findByText("Queryable report")

    fireEvent.change(screen.getByLabelText("Search shared sources"), {
      target: { value: "interview" }
    })
    await waitFor(() =>
      expect(api.listSources).toHaveBeenCalledWith(
        42,
        expect.objectContaining({ q: "interview", offset: 0 }),
        expect.any(AbortSignal)
      )
    )

    fireEvent.change(screen.getByLabelText("Filter shared sources by state"), {
      target: { value: "processing" }
    })
    await waitFor(() =>
      expect(api.listSources).toHaveBeenCalledWith(
        42,
        expect.objectContaining({ state: "processing", offset: 0 }),
        expect.any(AbortSignal)
      )
    )

    fireEvent.click(screen.getByRole("button", { name: "Next source page" }))
    await waitFor(() =>
      expect(api.listSources).toHaveBeenCalledWith(
        42,
        expect.objectContaining({ offset: 100 }),
        expect.any(AbortSignal)
      )
    )
    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeChecked()
  })

  it("seeds the exact server default, persists a turn, and opens citation evidence", async () => {
    renderWorkspace()

    await waitFor(() =>
      expect(screen.getByTestId("model-selector")).toHaveTextContent(
        "Anthropic / claude-shared"
      )
    )
    expect(fetchChatModels).toHaveBeenCalledWith({ returnEmpty: true })

    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "What does the report conclude?" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))

    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(1))
    const request = api.ask.mock.calls[0][1]
    expect(request).toMatchObject({
      request_id: expect.stringMatching(/^[0-9a-f-]{36}$/i),
      provider: "anthropic",
      model: "claude-shared",
      source_scope: { mode: "all", source_ids: [] }
    })
    await waitFor(() =>
      expect(
        screen.getByRole("log", { name: "Shared workspace messages" })
      ).toHaveTextContent("The report supports one conclusion.")
    )
    expect(screen.getByLabelText("Ask about shared sources")).toHaveValue("")

    const citation = screen.getByRole("button", {
      name: "Open citation 1 from Queryable report"
    })
    expect(citation).toHaveTextContent("Evidence from the report.")
    fireEvent.click(citation)
    await waitFor(() =>
      expect(api.previewSource).toHaveBeenCalledWith(
        42,
        "source-ready",
        7,
        expect.any(AbortSignal)
      )
    )
    expect(await screen.findByRole("dialog", { name: "Source preview" })).toHaveTextContent(
      "Chunk seven evidence"
    )
  })

  it("preserves the exact server provider identifier for a local generation default", async () => {
    api.bootstrap.mockResolvedValue(
      buildBootstrap({
        generation_default: {
          provider: "local-llm",
          model: "Qwen2.5-0.5B-Instruct",
          ready: true,
          reason_code: null
        }
      })
    )
    fetchChatModels.mockResolvedValue([])
    renderWorkspace()

    await waitFor(() =>
      expect(screen.getByTestId("model-selector")).toHaveTextContent(
        "local / Qwen2.5-0.5B-Instruct"
      )
    )
    fireEvent.change(screen.getByLabelText("Ask about shared sources"), {
      target: { value: "What does the report conclude?" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))

    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(1))
    expect(api.ask.mock.calls[0][1]).toMatchObject({
      provider: "local-llm",
      model: "Qwen2.5-0.5B-Instruct"
    })
  })

  it("loads older history upward without duplicate IDs or losing the scroll anchor", async () => {
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView
    renderWorkspace()
    await waitFor(() =>
      expect(
        screen.getByRole("log", { name: "Shared workspace messages" })
      ).toHaveTextContent("Existing grounded answer.")
    )
    const log = screen.getByRole("log", { name: "Shared workspace messages" })
    Object.defineProperty(log, "scrollTop", { value: 120, writable: true })
    const anchor = document.querySelector<HTMLElement>(
      '[data-message-id="message-existing"]'
    )
    expect(anchor).not.toBeNull()
    const rect = vi
      .spyOn(anchor!, "getBoundingClientRect")
      .mockReturnValueOnce({ top: 200 } as DOMRect)
      .mockReturnValueOnce({ top: 260 } as DOMRect)
    scrollIntoView.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Load older messages" }))
    await waitFor(() =>
      expect(
        screen.getByRole("log", { name: "Shared workspace messages" })
      ).toHaveTextContent("Older question")
    )
    expect(document.querySelectorAll('[data-message-id="message-existing"]')).toHaveLength(1)
    expect(log.scrollTop).toBe(180)
    expect(scrollIntoView).not.toHaveBeenCalled()
    rect.mockRestore()
  })

  it("renders history failure recovery and retries older messages", async () => {
    api.listMessages
      .mockRejectedValueOnce(
        new TldwApiError("history unavailable", 503, {
          code: "retrieval_unavailable",
          message: "Older messages are temporarily unavailable.",
          retryable: true
        })
      )
      .mockResolvedValueOnce({
        conversation_id: "conversation-1",
        messages: [
          {
            message_id: "message-older",
            role: "user",
            content: "Older question",
            created_at: "2026-08-20T10:00:00Z",
            citations: []
          }
        ],
        next_before: null
      })
    renderWorkspace()
    expect(
      await screen.findByRole("log", { name: "Shared workspace messages" })
    ).toHaveTextContent("Existing grounded answer.")

    fireEvent.click(screen.getByRole("button", { name: "Load older messages" }))
    expect(
      await screen.findByText("Older messages are temporarily unavailable.")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry older messages" }))

    expect(await screen.findByText("Older question")).toBeInTheDocument()
    expect(api.listMessages).toHaveBeenCalledTimes(2)
  })

  it("preserves draft and source scope across conflict and rate-limit recovery", async () => {
    api.ask
      .mockRejectedValueOnce(
        new TldwApiError("changed", 409, {
          code: "shared_source_changed",
          message: "changed",
          retryable: true,
          recovery_action: "refresh"
        })
      )
      .mockRejectedValueOnce(
        new TldwApiError("slow down", 429, {
          code: "shared_chat_rate_limited",
          message: "slow down",
          retryable: true,
          retry_after_ms: 1_000
        })
      )
    renderWorkspace()
    await screen.findByText("Queryable report")
    const composer = screen.getByLabelText("Ask about shared sources")

    fireEvent.change(composer, { target: { value: "Keep this draft" } })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))
    expect(
      await screen.findByText(
        "The shared source set changed. Refresh sources before trying again."
      )
    ).toBeInTheDocument()
    expect(composer).toHaveValue("Keep this draft")
    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeChecked()
    await waitFor(() => expect(api.listSources).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))
    expect(await screen.findByRole("status")).toHaveTextContent(/Try again in 1 second/)
    expect(api.ask.mock.calls[1][1].request_id).not.toBe(
      api.ask.mock.calls[0][1].request_id
    )
    expect(composer).toHaveValue("Keep this draft")
    expect(screen.getByRole("button", { name: "Ask shared workspace" })).toBeDisabled()
  })

  it.each([
    [
      "no_provider_configured",
      "Choose a configured model before asking a question."
    ],
    ["generation_failed", "Generation failed. Try again."],
    [
      "retrieval_unavailable",
      "Shared source retrieval is temporarily unavailable. Try again."
    ],
    [
      "shared_chat_context_too_large",
      "The selected sources exceed this model's context budget. Choose fewer sources and try again."
    ]
  ])("preserves recipient state for %s", async (code, copy) => {
    api.ask.mockRejectedValue(
      new TldwApiError("submission failed", 422, {
        code,
        message: "submission failed",
        retryable: false
      })
    )
    renderWorkspace()
    await screen.findByText("Queryable report")
    const composer = screen.getByLabelText("Ask about shared sources")
    fireEvent.change(composer, { target: { value: "Keep selected evidence" } })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))

    expect(await screen.findByText(copy)).toBeInTheDocument()
    expect(composer).toHaveValue("Keep selected evidence")
    expect(screen.getByRole("checkbox", { name: "Select Queryable report" })).toBeChecked()
  })

  it("fails closed when no generation target is available", async () => {
    api.bootstrap.mockResolvedValue(
      buildBootstrap({
        generation_default: {
          provider: null,
          model: null,
          ready: false,
          reason_code: "no_provider_configured"
        }
      })
    )
    renderWorkspace()

    expect(
      await screen.findByText("Choose a configured model before asking a question.")
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open model settings" })).toHaveAttribute(
      "href",
      "/settings/tldw"
    )
    expect(screen.getByRole("button", { name: "Ask shared workspace" })).toBeDisabled()
  })

  it("shows direct removed-source copy when citation evidence is withdrawn", async () => {
    api.previewSource.mockRejectedValue(
      new TldwApiError("removed", 404, {
        code: "shared_workspace_not_found",
        message: "removed",
        retryable: false
      })
    )
    renderWorkspace()
    fireEvent.change(await screen.findByLabelText("Ask about shared sources"), {
      target: { value: "Show the evidence" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Ask shared workspace" }))
    const citation = await screen.findByRole("button", {
      name: "Open citation 1 from Queryable report"
    })
    fireEvent.click(citation)

    expect(
      await screen.findByText("This source is no longer shared.")
    ).toBeInTheDocument()
  })

  it("fails closed for unavailable shares and offers only scoped recovery", async () => {
    api.bootstrap.mockRejectedValue(
      new TldwApiError("gone", 404, {
        code: "shared_workspace_not_found",
        message: "gone",
        retryable: false
      })
    )
    renderWorkspace()

    expect(
      await screen.findByRole("heading", {
        name: "This shared workspace isn't available."
      })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Return to Shared with me" })
    ).toHaveAttribute("href", "/shared-with-me")
    expect(screen.queryByText("Queryable report")).not.toBeInTheDocument()
  })
})
