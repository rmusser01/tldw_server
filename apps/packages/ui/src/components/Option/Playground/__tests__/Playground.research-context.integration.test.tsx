// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { Playground } from "../Playground"

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [],
    history: [],
    historyId: "history-1",
    serverChatId: "chat-1",
    serverChatTitle: "Research chat",
    serverChatLoadState: "loaded" as "idle" | "loading" | "loaded" | "failed",
    serverChatLoadError: null as string | null,
    serverChatState: "active" as string | null,
    serverChatTopic: null as string | null,
    serverChatSource: "webui" as string | null,
    isLoading: false,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    selectedSystemPrompt: null as string | null,
    setSelectedSystemPrompt: vi.fn(),
    selectedQuickPrompt: null as string | null,
    setSelectedQuickPrompt: vi.fn(),
    selectedModel: null as string | null,
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    contextFiles: [],
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
    selectedAssistant: null,
    setSelectedAssistant: vi.fn(),
    serverChatPersonaMemoryMode: null as "read_only" | "read_write" | null,
    temporaryChat: false,
    webSearch: false,
    toolChoice: "none" as const,
    setToolChoice: vi.fn(),
    selectedKnowledge: null,
    setSelectedKnowledge: vi.fn(),
    ragMediaIds: null as number[] | null,
    setRagMediaIds: vi.fn(),
    stopStreamingRequest: vi.fn(),
    regenerateLastMessage: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false
  }
}))

const artifactsState = vi.hoisted(() => ({
  value: {
    isOpen: false,
    active: null,
    isPinned: false,
    history: [],
    unreadCount: 0,
    setOpen: vi.fn(),
    closeArtifact: vi.fn(),
    markRead: vi.fn()
  }
}))

const smartScrollState = vi.hoisted(() => ({
  value: {
    containerRef: { current: null } as React.MutableRefObject<HTMLDivElement | null>,
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn()
  }
}))

const mobileViewportState = vi.hoisted(() => ({
  value: false
}))

const storeOptionState = vi.hoisted(() => ({
  value: {
    compareParentByHistory: {} as Record<
      string,
      { parentHistoryId: string; clusterId?: string }
    >,
    setSelectedQuickPrompt: vi.fn()
  }
}))

const chatSettingsState = vi.hoisted(() => ({
  syncChatSettingsForServerChat: vi.fn(async () => null),
  applyChatSettingsPatch: vi.fn(async () => null)
}))

const researchClientMocks = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue(undefined),
  getResearchBundle: vi.fn().mockResolvedValue({
    question: "Prepared follow-up question",
    outline: { sections: [{ title: "Overview" }] },
    claims: [{ text: "Claim one" }],
    unresolved_questions: ["Open question"],
    verification_summary: { unsupported_claim_count: 0 },
    source_trust: [{ source_id: "src_1", trust_tier: "high" }]
  })
}))

const buildAttachedContext = (runId: string, query: string) => ({
  attached_at: "2026-03-08T20:00:00Z",
  run_id: runId,
  query,
  question: query,
  outline: [{ title: "Overview" }],
  key_claims: [{ text: "Claim one" }],
  unresolved_questions: ["Open question"],
  verification_summary: { unsupported_claim_count: 0 },
  source_trust_summary: { high_trust_count: 1 },
  research_url: `/research?run=${runId}`
})

const buildPersistedAttachment = (runId: string, query: string) => ({
  ...buildAttachedContext(runId, query),
  updatedAt: "2026-03-08T20:05:00Z"
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string, options?: Record<string, unknown>) => {
      const template = defaultValue || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: (props: {
    attachedResearchContext?: {
      run_id?: string
      query?: string
      question?: string
    } | null
    attachedResearchContextBaseline?: {
      run_id?: string
      query?: string
      question?: string
    } | null
    attachedResearchContextPinned?: {
      run_id?: string
      query?: string
      question?: string
    } | null
    attachedResearchContextHistory?: Array<{
      run_id?: string
      query?: string
      question?: string
    }>
    onApplyAttachedResearchContext?: (
      context: ReturnType<typeof buildAttachedContext>
    ) => void
    onResetAttachedResearchContext?: () => void
    onRemoveAttachedResearchContext?: () => void
    onPinAttachedResearchContext?: () => void
    onUnpinAttachedResearchContext?: () => void
    onRestorePinnedResearchContext?: () => void
    onPrepareResearchFollowUp?: (target: { run_id: string; query: string }) => void
    onPinAttachedResearchContextHistory?: (
      context: ReturnType<typeof buildAttachedContext>
    ) => void
    onSelectAttachedResearchContextHistory?: (
      context: ReturnType<typeof buildAttachedContext>
    ) => void
  }) => {
    const [pendingFollowUp, setPendingFollowUp] = React.useState<{
      run_id: string
      query: string
    } | null>(null)

    return (
      <div
        data-testid="playground-form"
        data-attached-run-id={props.attachedResearchContext?.run_id || ""}
        data-attached-query={props.attachedResearchContext?.query || ""}
        data-attached-question={props.attachedResearchContext?.question || ""}
        data-baseline-run-id={props.attachedResearchContextBaseline?.run_id || ""}
        data-baseline-question={props.attachedResearchContextBaseline?.question || ""}
        data-pinned-run-id={props.attachedResearchContextPinned?.run_id || ""}
        data-history-run-ids={(props.attachedResearchContextHistory || [])
          .map((entry) => entry.run_id || "")
          .join(",")}
      >
        {props.attachedResearchContext ? (
          <div data-testid="active-attachment-surface">
            <button
              type="button"
              onClick={() =>
                props.onApplyAttachedResearchContext?.({
                  ...buildAttachedContext(
                    props.attachedResearchContext?.run_id || "run_edited",
                    props.attachedResearchContext?.query || "Edited query"
                  ),
                  question: "Edited attached question"
                })
              }
            >
              Edit attached research
            </button>
            <button
              type="button"
              onClick={() => props.onResetAttachedResearchContext?.()}
            >
              Reset attached research
            </button>
            <button
              type="button"
              onClick={() => props.onRemoveAttachedResearchContext?.()}
            >
              Remove attached research
            </button>
            <button
              type="button"
              onClick={() => props.onPinAttachedResearchContext?.()}
            >
              Pin attached research
            </button>
            {props.onPrepareResearchFollowUp ? (
              <button
                type="button"
                onClick={() =>
                  setPendingFollowUp({
                    run_id: props.attachedResearchContext?.run_id || "run_attached",
                    query: props.attachedResearchContext?.query || "Attached query"
                  })
                }
              >
                Follow up
              </button>
            ) : null}
          </div>
        ) : null}
        {props.attachedResearchContextPinned ? (
          <div data-testid="pinned-attachment-surface">
            <button
              type="button"
              onClick={() => props.onRestorePinnedResearchContext?.()}
            >
              Restore pinned research
            </button>
            <button
              type="button"
              onClick={() => props.onUnpinAttachedResearchContext?.()}
            >
              Unpin attached research
            </button>
            {props.onPrepareResearchFollowUp ? (
              <button
                type="button"
                onClick={() =>
                  setPendingFollowUp({
                    run_id: props.attachedResearchContextPinned?.run_id || "run_pinned",
                    query: props.attachedResearchContextPinned?.query || "Pinned query"
                  })
                }
              >
                Follow up
              </button>
            ) : null}
          </div>
        ) : null}
        {(props.attachedResearchContextHistory || []).map((entry) => (
          <div
            key={entry.run_id}
            data-testid={`history-attachment-surface-${entry.run_id}`}
          >
            <button
              type="button"
              onClick={() =>
                props.onSelectAttachedResearchContextHistory?.(
                  buildAttachedContext(
                    entry.run_id || "run_history",
                    entry.query || "History query"
                  )
                )
              }
            >
              {`Use history ${entry.run_id}`}
            </button>
            <button
              type="button"
              onClick={() =>
                props.onPinAttachedResearchContextHistory?.(
                  buildAttachedContext(
                    entry.run_id || "run_history",
                    entry.query || "History query"
                  )
                )
              }
            >
              {`Pin history ${entry.run_id}`}
            </button>
            {props.onPrepareResearchFollowUp ? (
              <button
                type="button"
                onClick={() =>
                  setPendingFollowUp({
                    run_id: entry.run_id || "run_history",
                    query: entry.query || "History query"
                  })
                }
              >
                Follow up
              </button>
            ) : null}
          </div>
        ))}
        {pendingFollowUp ? (
          <div data-testid="follow-up-confirmation">
            <div>Prepare follow-up?</div>
            <div>
              {`This will use "${pendingFollowUp.query}" and prefill a follow-up research prompt in the composer.`}
            </div>
            <button
              type="button"
              onClick={() => {
                props.onPrepareResearchFollowUp?.(pendingFollowUp)
                setPendingFollowUp(null)
              }}
            >
              Prepare follow-up
            </button>
            <button
              type="button"
              onClick={() => setPendingFollowUp(null)}
            >
              Cancel
            </button>
          </div>
        ) : null}
      </div>
    )
  }
}))

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: (props: {
    onAttachResearchContext?: (context: ReturnType<typeof buildAttachedContext>) => void
    onPrepareResearchFollowUp?: (target: { run_id: string; query: string }) => void
    returnedResearchRunId?: string | null
    onDismissReturnedResearchRun?: () => void
  }) => (
    <div data-testid="playground-chat">
      {props.returnedResearchRunId ? (
        <div
          data-testid="returned-research-banner"
          data-returned-run-id={props.returnedResearchRunId}
        >
          <button
            type="button"
            onClick={() => props.onDismissReturnedResearchRun?.()}
          >
            Dismiss returned research
          </button>
        </div>
      ) : null}
      <button
        type="button"
        onClick={() =>
          props.onAttachResearchContext?.(
            buildAttachedContext("run_1", "Battery recycling supply chain")
          )
        }
      >
        Attach run 1
      </button>
      <button
        type="button"
        onClick={() =>
          props.onAttachResearchContext?.(
            buildAttachedContext("run_2", "Grid-scale recycling economics")
          )
        }
      >
        Attach run 2
      </button>
      <button
        type="button"
        onClick={() =>
          props.onPrepareResearchFollowUp?.({
            run_id: "run_follow_up",
            query: "Battery recycling supply chain"
          })
        }
      >
        Prepare follow up
      </button>
    </div>
  )
}))

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => <div data-testid="artifacts-panel" />
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState.value
}))

vi.mock("@/hooks/usePlaygroundSessionPersistence", () => ({
  usePlaygroundSessionPersistence: () => ({
    restoreSession: vi.fn(async () => false),
    sessionScopeReady: true,
    hasPersistedSession: false,
    persistedHistoryId: null,
    persistedServerChatId: null
  })
}))

vi.mock("@/hooks/playground-session-restore", () => ({
  shouldRestorePersistedPlaygroundSession: () => false
}))

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
}))

vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getPromptById: vi.fn(async () => null),
  getRecentChatFromWebUI: vi.fn(async () => null)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ setSystemPrompt: vi.fn() })
}))

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => smartScrollState.value
}))

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage"
}))

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: []
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: typeof storeOptionState.value) => unknown
  ) => selector(storeOptionState.value)
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (selector: (state: typeof artifactsState.value) => unknown) =>
    selector(artifactsState.value)
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [""]
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => mobileViewportState.value
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: (...args: unknown[]) =>
    chatSettingsState.syncChatSettingsForServerChat(...args),
  applyChatSettingsPatch: (...args: unknown[]) =>
    chatSettingsState.applyChatSettingsPatch(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: researchClientMocks
}))

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => vi.fn(async () => {})
}))

vi.mock("../playground-shortcuts", () => ({
  resolvePlaygroundShortcutAction: () => null
}))

vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: () => undefined
}))

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useNavigate: () => vi.fn()
  }
})

describe("Playground research context integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.history.replaceState({}, "", "/chat")
    mobileViewportState.value = false
    artifactsState.value.isOpen = false
    storeOptionState.value.compareParentByHistory = {}
    storeOptionState.value.setSelectedQuickPrompt = vi.fn()
    messageOptionState.value.messages = []
    messageOptionState.value.history = []
    messageOptionState.value.serverChatId = "chat-1"
    messageOptionState.value.serverChatTitle = "Research chat"
    messageOptionState.value.serverChatLoadState = "loaded"
    messageOptionState.value.serverChatLoadError = null
    messageOptionState.value.serverChatState = "active"
    messageOptionState.value.serverChatTopic = null
    messageOptionState.value.serverChatSource = "webui"
    messageOptionState.value.historyId = "history-1"
    messageOptionState.value.selectedSystemPrompt = null
    messageOptionState.value.selectedQuickPrompt = null
    messageOptionState.value.setSelectedQuickPrompt =
      storeOptionState.value.setSelectedQuickPrompt
    messageOptionState.value.contextFiles = []
    messageOptionState.value.selectedKnowledge = null
    messageOptionState.value.ragMediaIds = null
    messageOptionState.value.selectedCharacter = null
    messageOptionState.value.selectedAssistant = null
    messageOptionState.value.temporaryChat = false
    messageOptionState.value.webSearch = false
    messageOptionState.value.toolChoice = "none"
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue(null)
    chatSettingsState.applyChatSettingsPatch.mockResolvedValue(null)
  })

  it("prepares follow-up research by attaching the selected run and seeding the deterministic draft", async () => {
    render(<Playground />)

    fireEvent.click(screen.getByRole("button", { name: "Prepare follow up" }))

    await waitFor(() =>
      expect(researchClientMocks.getResearchBundle).toHaveBeenCalledWith(
        "run_follow_up"
      )
    )
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_follow_up"
      )
    )
    expect(storeOptionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(
      "Follow up on this research: Battery recycling supply chain"
    )
  })

  it("consumes researchReturnRunId once and passes the returned run marker to PlaygroundChat", async () => {
    window.history.replaceState(
      {},
      "",
      "/chat?settingsServerChatId=chat-1&researchReturnRunId=run_returned"
    )

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("returned-research-banner")).toHaveAttribute(
        "data-returned-run-id",
        "run_returned"
      )
    )
    await waitFor(() =>
      expect(window.location.search).toBe("")
    )
  })

  it("shows Follow up on the active attachment surface and opens the local confirmation before preparing follow-up research", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_active",
        "Active battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1")
      ]
    })

    render(<Playground />)

    const followUpButton = await within(
      await screen.findByTestId("active-attachment-surface")
    ).findByRole("button", { name: "Follow up" })
    fireEvent.click(followUpButton)

    expect(
      screen.getByText("Prepare follow-up?")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Prepare follow-up" })
    ).toBeInTheDocument()
  })

  it("shows Follow up on the pinned mini-card and keeps the confirmation local", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: null,
      deepResearchPinnedAttachment: buildPersistedAttachment(
        "run_pinned",
        "Pinned battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1")
      ]
    })

    render(<Playground />)

    const followUpButton = await within(
      await screen.findByTestId("pinned-attachment-surface")
    ).findByRole("button", { name: "Follow up" })
    fireEvent.click(followUpButton)

    expect(
      screen.getByText("Prepare follow-up?")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Cancel" })
    ).toBeInTheDocument()
  })

  it("shows Follow up on recent-history entries and cancels without changing the active attachment", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_active",
        "Active battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1"),
        buildPersistedAttachment("run_hist_2", "History 2")
      ]
    })

    render(<Playground />)

    const followUpButton = await within(
      await screen.findByTestId("history-attachment-surface-run_hist_1")
    ).findByRole("button", { name: "Follow up" })
    fireEvent.click(followUpButton)

    expect(
      screen.getByText("Prepare follow-up?")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    expect(screen.queryByText("Prepare follow-up?")).not.toBeInTheDocument()
    expect(researchClientMocks.getResearchBundle).not.toHaveBeenCalled()
    expect(storeOptionState.value.setSelectedQuickPrompt).not.toHaveBeenCalled()
  })

  it("confirming follow-up does not implicitly trigger Use", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_active",
        "Active battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1")
      ]
    })

    render(<Playground />)

    const followUpButton = await within(
      await screen.findByTestId("active-attachment-surface")
    ).findByRole("button", { name: "Follow up" })
    fireEvent.click(followUpButton)
    fireEvent.click(screen.getByRole("button", { name: "Prepare follow-up" }))

    expect(researchClientMocks.getResearchBundle).not.toHaveBeenCalled()
    expect(storeOptionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(
      "Follow up on this research: Active battery recycling run"
    )
    expect(
      screen.getByRole("button", { name: "Use history run_hist_1" })
    ).toBeInTheDocument()
  })

  it("replaces the attached research context and lets the form remove it", async () => {
    render(<Playground />)

    fireEvent.click(screen.getByRole("button", { name: "Attach run 1" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_1"
      )
    )

    fireEvent.click(screen.getByRole("button", { name: "Attach run 2" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_2"
      )
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Remove attached research" })
    )
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        ""
      )
    )
  })

  it("tracks a run-derived baseline and lets preview edits update only the active attachment", async () => {
    render(<Playground />)

    fireEvent.click(screen.getByRole("button", { name: "Attach run 1" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_1"
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-attached-question",
      "Battery recycling supply chain"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-run-id",
      "run_1"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-question",
      "Battery recycling supply chain"
    )

    fireEvent.click(screen.getByRole("button", { name: "Edit attached research" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-question",
        "Edited attached question"
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-question",
      "Battery recycling supply chain"
    )

    fireEvent.click(screen.getByRole("button", { name: "Reset attached research" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-question",
        "Battery recycling supply chain"
      )
    )
  })

  it("clears attached research context when the active thread changes", async () => {
    const view = render(<Playground />)

    fireEvent.click(screen.getByRole("button", { name: "Attach run 1" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_1"
      )
    )

    messageOptionState.value.serverChatId = "chat-2"
    view.rerender(<Playground />)
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        ""
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-run-id",
      ""
    )

    fireEvent.click(screen.getByRole("button", { name: "Attach run 2" }))
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_2"
      )
    )

    messageOptionState.value.historyId = "history-2"
    view.rerender(<Playground />)
    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        ""
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-run-id",
      ""
    )
  })

  it("auto-restores persisted attached research context and bounded history for saved chats", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_saved",
        "Recovered battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1"),
        buildPersistedAttachment("run_hist_2", "History 2")
      ]
    })

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_saved"
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-run-id",
      "run_saved"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      ""
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-history-run-ids",
      "run_hist_1,run_hist_2"
    )
    expect(chatSettingsState.syncChatSettingsForServerChat).toHaveBeenCalledWith({
      historyId: "history-1",
      serverChatId: "chat-1"
    })
  })

  it("auto-restores the pinned attachment when a saved chat has no active attachment", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchPinnedAttachment: buildPersistedAttachment(
        "run_pinned",
        "Pinned recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1")
      ]
    })

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_pinned"
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-baseline-run-id",
      "run_pinned"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      "run_pinned"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-history-run-ids",
      "run_hist_1"
    )
  })

  it("persists attachment attach, edit, remove, and history swaps for saved chats", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_saved",
        "Recovered battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1"),
        buildPersistedAttachment("run_hist_2", "History 2")
      ]
    })

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_saved"
      )
    )

    fireEvent.click(screen.getByRole("button", { name: "Attach run 1" }))
    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          historyId: "history-1",
          serverChatId: "chat-1",
          patch: expect.objectContaining({
            deepResearchAttachment: expect.objectContaining({
              run_id: "run_1",
              query: "Battery recycling supply chain"
            }),
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_saved" }),
              expect.objectContaining({ run_id: "run_hist_1" }),
              expect.objectContaining({ run_id: "run_hist_2" })
            ]
          })
        })
      )
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Edit attached research" })
    )
    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchAttachment: expect.objectContaining({
              question: "Edited attached question"
            }),
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_saved" }),
              expect.objectContaining({ run_id: "run_hist_1" }),
              expect.objectContaining({ run_id: "run_hist_2" })
            ]
          })
        })
      )
    )

    fireEvent.click(screen.getByRole("button", { name: "Use history run_hist_1" }))
    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchAttachment: expect.objectContaining({
              run_id: "run_hist_1"
            }),
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_1" }),
              expect.objectContaining({ run_id: "run_saved" }),
              expect.objectContaining({ run_id: "run_hist_2" })
            ]
          })
        })
      )
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Remove attached research" })
    )
    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchAttachment: null,
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_1" }),
              expect.objectContaining({ run_id: "run_saved" }),
              expect.objectContaining({ run_id: "run_hist_2" })
            ]
          })
        })
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-history-run-ids",
      "run_1,run_saved,run_hist_2"
    )
    expect(
      screen.getByRole("button", { name: "Use history run_1" })
    ).toBeInTheDocument()
  })

  it("persists pin and unpin actions for saved chats", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_saved",
        "Recovered battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1")
      ]
    })

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_saved"
      )
    )

    fireEvent.click(screen.getByRole("button", { name: "Pin attached research" }))
    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchPinnedAttachment: expect.objectContaining({
              run_id: "run_saved"
            }),
            deepResearchAttachment: expect.objectContaining({
              run_id: "run_saved"
            }),
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_hist_1" })
            ]
          })
        })
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      "run_saved"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Unpin attached research" })
    )
    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchPinnedAttachment: null,
            deepResearchAttachment: expect.objectContaining({
              run_id: "run_saved"
            })
          })
        })
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      ""
    )
  })

  it("pins a history entry without changing the active attachment", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: buildPersistedAttachment(
        "run_saved",
        "Recovered battery recycling run"
      ),
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1"),
        buildPersistedAttachment("run_hist_2", "History 2")
      ]
    })

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_saved"
      )
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Pin history run_hist_1" })
    )

    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchPinnedAttachment: expect.objectContaining({
              run_id: "run_hist_1"
            }),
            deepResearchAttachment: expect.objectContaining({
              run_id: "run_saved"
            }),
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_hist_2" })
            ]
          })
        })
      )
    )

    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-attached-run-id",
      "run_saved"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      "run_hist_1"
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-history-run-ids",
      "run_hist_2"
    )
  })

  it("pins a history entry from the fallback surface when no active attachment exists", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockResolvedValue({
      updatedAt: "2026-03-08T20:10:00Z",
      deepResearchAttachment: null,
      deepResearchAttachmentHistory: [
        buildPersistedAttachment("run_hist_1", "History 1"),
        buildPersistedAttachment("run_hist_2", "History 2")
      ]
    })

    render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        ""
      )
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Pin history run_hist_1" })
    )

    await waitFor(() =>
      expect(chatSettingsState.applyChatSettingsPatch).toHaveBeenCalledWith(
        expect.objectContaining({
          patch: expect.objectContaining({
            deepResearchPinnedAttachment: expect.objectContaining({
              run_id: "run_hist_1"
            }),
            deepResearchAttachment: null,
            deepResearchAttachmentHistory: [
              expect.objectContaining({ run_id: "run_hist_2" })
            ]
          })
        })
      )
    )

    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-attached-run-id",
      ""
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      "run_hist_1"
    )
  })

  it("does not persist attachments for temporary chats", async () => {
    messageOptionState.value.serverChatId = null
    messageOptionState.value.historyId = "temp"

    render(<Playground />)

    fireEvent.click(screen.getByRole("button", { name: "Attach run 1" }))

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_1"
      )
    )
    expect(chatSettingsState.syncChatSettingsForServerChat).not.toHaveBeenCalled()
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled()
  })

  it("restores the correct persisted attachment when switching saved chats", async () => {
    chatSettingsState.syncChatSettingsForServerChat.mockImplementation(
      async ({ serverChatId }: { serverChatId: string }) => {
        if (serverChatId === "chat-1") {
          return {
            updatedAt: "2026-03-08T20:10:00Z",
            deepResearchAttachment: buildPersistedAttachment(
              "run_one",
              "First saved run"
            ),
            deepResearchPinnedAttachment: buildPersistedAttachment(
              "run_one_pinned",
              "First saved pinned run"
            )
          }
        }
        if (serverChatId === "chat-2") {
          return {
            updatedAt: "2026-03-08T20:11:00Z",
            deepResearchAttachment: buildPersistedAttachment(
              "run_two",
              "Second saved run"
            ),
            deepResearchPinnedAttachment: buildPersistedAttachment(
              "run_two_pinned",
              "Second saved pinned run"
            )
          }
        }
        return null
      }
    )

    const view = render(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_one"
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      "run_one_pinned"
    )

    messageOptionState.value.serverChatId = "chat-2"
    messageOptionState.value.historyId = "history-2"
    view.rerender(<Playground />)

    await waitFor(() =>
      expect(screen.getByTestId("playground-form")).toHaveAttribute(
        "data-attached-run-id",
        "run_two"
      )
    )
    expect(screen.getByTestId("playground-form")).toHaveAttribute(
      "data-pinned-run-id",
      "run_two_pinned"
    )
  })
})
