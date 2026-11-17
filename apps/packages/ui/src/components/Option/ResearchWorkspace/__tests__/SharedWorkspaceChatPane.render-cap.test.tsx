// Stage 3 (WP-07) bounded growth for the shared workspace transcript: the
// pane caps rendered history to the newest K=200 messages with a client-side
// "Show older messages" window reveal, alongside the existing server-side
// "Load older messages" pagination.
import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  SHARED_WORKSPACE_CHAT_RENDER_WINDOW,
  SharedWorkspaceChatPane
} from "../SharedResearchWorkspace/SharedWorkspaceChatPane"
import type { SharedResearchWorkspaceController } from "../SharedResearchWorkspace/useSharedResearchWorkspace"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/components/Option/Playground/ChatModelSelectorDropdown", () => ({
  ChatModelSelectorDropdown: () => null
}))

vi.mock("@/hooks/playground", () => ({
  useModelSelector: () => ({
    apiModelLabel: "Test Model",
    modelDropdownMenuItems: [],
    modelDropdownOpen: false,
    setModelDropdownOpen: vi.fn(),
    modelSearchQuery: "",
    setModelSearchQuery: vi.fn(),
    modelSelectorWarning: null,
    modelSortMode: "model",
    setModelSortMode: vi.fn(),
    resolvedProviderKey: "test"
  })
}))

vi.mock("../SharedResearchWorkspace/SharedWorkspaceSafeMarkdown", () => ({
  SharedWorkspaceSafeMarkdown: ({ content }: { content: string }) => (
    <div data-testid="shared-markdown">{content}</div>
  )
}))

type SharedMessageFixture = {
  message_id: string
  role: "user" | "assistant"
  content: string
  citations: Array<Record<string, unknown>>
}

const buildSharedMessages = (count: number): SharedMessageFixture[] =>
  Array.from({ length: count }, (_unused, index) => ({
    message_id: `sm-${index}`,
    role: index % 2 === 1 ? "assistant" : "user",
    content: `shared content ${index}`,
    citations: []
  }))

const buildController = (
  messages: SharedMessageFixture[],
  extraState: Record<string, unknown> = {}
) =>
  ({
    setModel: vi.fn(),
    setProvider: vi.fn(),
    loadOlderHistory: vi.fn(async () => {}),
    submitDraft: vi.fn(async () => {}),
    retryPending: vi.fn(async () => {}),
    refreshSources: vi.fn(async () => {}),
    state: {
      draft: "",
      provider: "test",
      model: "test-model",
      selectionMaterializing: false,
      pendingSubmission: null,
      rateLimitRemainingMs: 0,
      allowedActions: {
        ask_grounded_questions: { allowed: true, reason_code: null },
        inspect_sources: { allowed: true, reason_code: null }
      },
      errors: { submission: null, history: null, bootstrap: null },
      bootstrap: null,
      sourceScopeMode: "all",
      sourceSummary: { queryable: 3 },
      selectedSourceIds: [],
      messages,
      lastCompletedAssistantMessageId: null,
      nextBefore: null,
      ...extraState
    }
  }) as unknown as SharedResearchWorkspaceController

const renderedArticleCount = () =>
  document.querySelectorAll("article[data-message-id]").length

const renderPane = (controller: SharedResearchWorkspaceController) =>
  render(
    <MemoryRouter>
      <SharedWorkspaceChatPane
        controller={controller}
        onPreviewCitation={vi.fn()}
      />
    </MemoryRouter>
  )

describe("SharedWorkspaceChatPane rendered-history cap (Stage 3, WP-07)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("exports the K=200 render window constant", () => {
    expect(SHARED_WORKSPACE_CHAT_RENDER_WINDOW).toBe(200)
  })

  it("caps rendering to the newest K=200 messages of a long transcript", () => {
    renderPane(buildController(buildSharedMessages(1000)))

    expect(renderedArticleCount()).toBe(SHARED_WORKSPACE_CHAT_RENDER_WINDOW)
    // Newest window: the newest message renders, messages older than the
    // newest K do not.
    expect(
      document.querySelector('article[data-message-id="sm-999"]')
    ).not.toBeNull()
    expect(
      document.querySelector('article[data-message-id="sm-800"]')
    ).not.toBeNull()
    expect(
      document.querySelector('article[data-message-id="sm-799"]')
    ).toBeNull()
  })

  it("reveals the next older window on Show older messages", () => {
    renderPane(buildController(buildSharedMessages(1000)))

    fireEvent.click(
      screen.getByRole("button", { name: /show older messages/i })
    )

    expect(renderedArticleCount()).toBe(SHARED_WORKSPACE_CHAT_RENDER_WINDOW * 2)
    expect(
      document.querySelector('article[data-message-id="sm-600"]')
    ).not.toBeNull()
    expect(
      document.querySelector('article[data-message-id="sm-599"]')
    ).toBeNull()
  })

  it("keeps the newest messages rendered as new answers arrive", () => {
    const controller = buildController(buildSharedMessages(1000))
    const page = renderPane(controller)

    const grown = buildSharedMessages(1001)
    page.rerender(
      <MemoryRouter>
        <SharedWorkspaceChatPane
          controller={buildController(grown)}
          onPreviewCitation={vi.fn()}
        />
      </MemoryRouter>
    )

    expect(renderedArticleCount()).toBe(SHARED_WORKSPACE_CHAT_RENDER_WINDOW)
    expect(
      document.querySelector('article[data-message-id="sm-1000"]')
    ).not.toBeNull()
  })

  it("keeps the server-side Load older messages affordance for unfetched history", () => {
    renderPane(
      buildController(buildSharedMessages(50), { nextBefore: "cursor-1" })
    )

    // Short transcripts render fully; the server pagination affordance is
    // unchanged by the render cap.
    expect(renderedArticleCount()).toBe(50)
    expect(
      screen.getByRole("button", { name: /load older messages/i })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /show older messages/i })
    ).not.toBeInTheDocument()
  })
})
