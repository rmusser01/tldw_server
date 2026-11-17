// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"
import { DemoModeProvider } from "@/context/demo-mode"

const useMessageOptionState = vi.hoisted(() => ({
  value: {
    messages: [] as any[],
    setMessages: vi.fn(),
    streaming: false,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    isSearchingInternet: false,
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    toggleMessagePinned: vi.fn(),
    ttsEnabled: false,
    onSubmit: vi.fn(),
    actionInfo: null,
    messageSteeringMode: "none",
    setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false,
    setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(),
    createChatBranch: vi.fn(),
    createCompareBranch: vi.fn(),
    temporaryChat: false,
    serverChatId: "chat-1",
    serverChatCharacterId: null,
    stopStreamingRequest: vi.fn(),
    isEmbedding: false,
    compareMode: false,
    compareFeatureEnabled: false,
    compareSelectionByCluster: {},
    setCompareSelectionForCluster: vi.fn(),
    compareActiveModelsByCluster: {},
    setCompareActiveModelsForCluster: vi.fn(),
    setCompareSelectedModels: vi.fn(),
    historyId: "history-1",
    setSelectedModel: vi.fn(),
    setCompareMode: vi.fn(),
    sendPerModelReply: vi.fn(),
    compareCanonicalByCluster: {},
    setCompareCanonicalForCluster: vi.fn(),
    compareContinuationModeByCluster: {},
    setCompareContinuationModeForCluster: vi.fn(),
    setCompareParentForHistory: vi.fn(),
    compareSplitChats: {},
    setCompareSplitChat: vi.fn(),
    compareMaxModels: 3
  }
}))

const queryState = vi.hoisted(() => ({
  linkedRuns: [] as any[]
}))

const clientMocks = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue(undefined),
  getResearchBundle: vi.fn().mockResolvedValue({
    question: "What changed in the battery recycling market?",
    outline: { sections: [{ title: "Overview" }] },
    claims: [{ text: "Claim one" }],
    unresolved_questions: ["What changed in Europe?"],
    verification_summary: { unsupported_claim_count: 0 },
    source_trust: [{ source_id: "src_1", trust_tier: "high" }]
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const queryClient = new actual.QueryClient()
  return {
    ...actual,
    useQueryClient: () => queryClient,
    useQuery: (options: Record<string, any>) => {
      const key = Array.isArray(options.queryKey) ? options.queryKey[0] : null
      if (key === "playground:chat-linked-research-runs") {
        return {
          data: { runs: queryState.linkedRuns },
          isSuccess: true,
          isError: false,
          status: "success",
          errorUpdatedAt: 0,
          dataUpdatedAt: 1
        }
      }
      return {
        data: [],
        isSuccess: true,
        isError: false,
        status: "success",
        errorUpdatedAt: 0,
        dataUpdatedAt: 1
      }
    }
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => useMessageOptionState.value
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => null
}))

vi.mock("./PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => null
}))

vi.mock("../PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => null
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    message: string
    researchActions?: {
      reasonLabel?: string
      primaryLink?: {
        href: string
        label: string
      }
      onUseInChat?: () => void
      onFollowUp?: () => void
    }
  }) => (
    <div data-testid="playground-message-mock">
      <div>{props.message}</div>
      {props.researchActions?.reasonLabel && (
        <div>{props.researchActions.reasonLabel}</div>
      )}
      {props.researchActions?.primaryLink && (
        <a href={props.researchActions.primaryLink.href}>
          {props.researchActions.primaryLink.label}
        </a>
      )}
      {props.researchActions?.onUseInChat && (
        <button type="button" onClick={() => props.researchActions?.onUseInChat?.()}>
          Use in Chat
        </button>
      )}
      {props.researchActions?.onFollowUp && (
        <button type="button" onClick={() => props.researchActions?.onFollowUp?.()}>
          Follow up
        </button>
      )}
    </div>
  )
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: clientMocks
}))

const renderChat = (extraProps?: Record<string, unknown>) =>
  render(
    <DemoModeProvider>
      <PlaygroundChat {...(extraProps as any)} />
    </DemoModeProvider>
  )

describe("PlaygroundChat research use-in-chat message integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    queryState.linkedRuns = []
    useMessageOptionState.value.messages = []
    useMessageOptionState.value.serverChatId = "chat-1"
    useMessageOptionState.value.historyId = "history-1"
  })

  const completionMessage = (
    runId: string,
    query: string,
    message = "Deep research finished."
  ) => ({
    isBot: true,
    role: "assistant",
    name: "Assistant",
    message,
    sources: [],
    metadataExtra: {
      deep_research_completion: {
        run_id: runId,
        query,
        kind: "completion_handoff"
      }
    }
  })

  it("shows Use in Chat for completion handoff messages and not for unrelated assistant messages", async () => {
    useMessageOptionState.value.messages = [
      {
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "Deep research finished.",
        sources: [],
        metadataExtra: {
          deep_research_completion: {
            run_id: "run_msg",
            query: "Battery recycling supply chain",
            kind: "completion_handoff"
          }
        }
      },
      {
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "Ordinary assistant reply.",
        sources: [],
        metadataExtra: {}
      }
    ]

    const attachSpy = vi.fn()
    renderChat({ onAttachResearchContext: attachSpy })

    const buttons = screen.getAllByRole("button", { name: "Use in Chat" })
    expect(buttons).toHaveLength(1)

    fireEvent.click(buttons[0])

    await waitFor(() => expect(clientMocks.getResearchBundle).toHaveBeenCalledWith("run_msg"))
    await waitFor(() =>
      expect(attachSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          run_id: "run_msg",
          query: "Battery recycling supply chain",
          research_url: "/research?run=run_msg"
        })
      )
    )
  })

  it("shows Follow up for completion handoff messages and not for unrelated assistant messages", async () => {
    useMessageOptionState.value.messages = [
      {
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "Deep research finished.",
        sources: [],
        metadataExtra: {
          deep_research_completion: {
            run_id: "run_msg",
            query: "Battery recycling supply chain",
            kind: "completion_handoff"
          }
        }
      },
      {
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "Ordinary assistant reply.",
        sources: [],
        metadataExtra: {}
      }
    ]

    const followUpSpy = vi.fn()
    renderChat({ onPrepareResearchFollowUp: followUpSpy })

    const buttons = screen.getAllByRole("button", { name: "Follow up" })
    expect(buttons).toHaveLength(1)

    fireEvent.click(buttons[0])

    await waitFor(() =>
      expect(followUpSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          run_id: "run_msg",
          query: "Battery recycling supply chain"
        })
      )
    )
  })

  it("shows checkpoint-aware review actions for plan review handoff messages", () => {
    queryState.linkedRuns = [
      {
        run_id: "run_msg",
        query: "Battery recycling supply chain",
        status: "waiting_human",
        phase: "awaiting_plan_review",
        control_state: "running",
        latest_checkpoint_id: "cp_1",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ]
    useMessageOptionState.value.messages = [
      completionMessage(
        "run_msg",
        "Battery recycling supply chain",
        "Deep research finished for plan review."
      )
    ]

    renderChat({ onAttachResearchContext: vi.fn(), onPrepareResearchFollowUp: vi.fn() })

    const message = within(
      screen.getByText("Deep research finished for plan review.").closest(
        '[data-testid="playground-message-mock"]'
      ) as HTMLElement
    )

    expect(message.getByText("Plan review needed")).toBeInTheDocument()
    expect(message.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=run_msg"
    )
    expect(message.queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(message.queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
  })

  it("shows checkpoint-aware review actions for sources and outline handoff messages", () => {
    queryState.linkedRuns = [
      {
        run_id: "run_sources",
        query: "Battery recycling sources",
        status: "waiting_human",
        phase: "awaiting_sources_review",
        control_state: "running",
        latest_checkpoint_id: "cp_sources",
        updated_at: "2026-03-19T20:03:00+00:00"
      },
      {
        run_id: "run_outline",
        query: "Battery recycling outline",
        status: "waiting_human",
        phase: "awaiting_outline_review",
        control_state: "running",
        latest_checkpoint_id: "cp_outline",
        updated_at: "2026-03-19T20:02:00+00:00"
      }
    ]
    useMessageOptionState.value.messages = [
      completionMessage(
        "run_sources",
        "Battery recycling sources",
        "Deep research finished for sources."
      ),
      completionMessage(
        "run_outline",
        "Battery recycling outline",
        "Deep research finished for outline."
      )
    ]

    renderChat()

    const sourcesMessage = within(
      screen.getByText("Deep research finished for sources.").closest(
        '[data-testid="playground-message-mock"]'
      ) as HTMLElement
    )
    const outlineMessage = within(
      screen.getByText("Deep research finished for outline.").closest(
        '[data-testid="playground-message-mock"]'
      ) as HTMLElement
    )

    expect(sourcesMessage.getByText("Sources review needed")).toBeInTheDocument()
    expect(outlineMessage.getByText("Outline review needed")).toBeInTheDocument()
    expect(sourcesMessage.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=run_sources"
    )
    expect(outlineMessage.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=run_outline"
    )
    expect(sourcesMessage.queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(sourcesMessage.queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
    expect(outlineMessage.queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(outlineMessage.queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
  })

  it("falls back to a generic review label for unknown waiting_human phases", () => {
    queryState.linkedRuns = [
      {
        run_id: "run_unknown",
        query: "Battery recycling unknown review",
        status: "waiting_human",
        phase: "awaiting_custom_review",
        control_state: "running",
        latest_checkpoint_id: "cp_custom",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ]
    useMessageOptionState.value.messages = [
      completionMessage(
        "run_unknown",
        "Battery recycling unknown review",
        "Deep research finished for unknown review."
      )
    ]

    renderChat()

    const message = within(
      screen.getByText("Deep research finished for unknown review.").closest(
        '[data-testid="playground-message-mock"]'
      ) as HTMLElement
    )

    expect(message.getByText("Review needed")).toBeInTheDocument()
    expect(message.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=run_unknown"
    )
    expect(message.queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(message.queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
  })

  it("keeps existing actions when no current linked-run match exists", () => {
    useMessageOptionState.value.messages = [
      completionMessage(
        "run_msg",
        "Battery recycling supply chain",
        "Deep research finished for no current linked run."
      )
    ]

    renderChat({ onAttachResearchContext: vi.fn(), onPrepareResearchFollowUp: vi.fn() })

    expect(screen.getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Follow up" })).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Review in Research" })).not.toBeInTheDocument()
  })

  it("leaves unrelated assistant messages without research handoff actions while a checkpoint run is active", () => {
    queryState.linkedRuns = [
      {
        run_id: "run_msg",
        query: "Battery recycling supply chain",
        status: "waiting_human",
        phase: "awaiting_plan_review",
        control_state: "running",
        latest_checkpoint_id: "cp_1",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ]
    useMessageOptionState.value.messages = [
      completionMessage(
        "run_msg",
        "Battery recycling supply chain",
        "Deep research finished for related handoff."
      ),
      {
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "Ordinary assistant reply.",
        sources: [],
        metadataExtra: {}
      }
    ]

    renderChat()

    const unrelated = within(
      screen.getByText("Ordinary assistant reply.").closest(
        '[data-testid="playground-message-mock"]'
      ) as HTMLElement
    )

    expect(unrelated.queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(unrelated.queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
    expect(unrelated.queryByRole("link", { name: "Review in Research" })).not.toBeInTheDocument()
  })

  it("refreshes cached row callbacks and review policy while message metadata stays identical", async () => {
    const message = completionMessage("run_msg", "Battery recycling supply chain")
    useMessageOptionState.value.messages = [message]
    const originalAttach = vi.fn()
    const originalFollowUp = vi.fn()
    const currentAttach = vi.fn()
    const currentFollowUp = vi.fn()
    const view = renderChat({
      onAttachResearchContext: originalAttach,
      onPrepareResearchFollowUp: originalFollowUp
    })
    view.rerender(
      <DemoModeProvider>
        <PlaygroundChat
          onAttachResearchContext={currentAttach}
          onPrepareResearchFollowUp={currentFollowUp}
        />
      </DemoModeProvider>
    )
    fireEvent.click(screen.getByRole("button", { name: "Use in Chat" }))
    await waitFor(() => expect(currentAttach).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole("button", { name: "Follow up" }))
    expect(currentFollowUp).toHaveBeenCalledTimes(1)
    expect(originalAttach).not.toHaveBeenCalled()
    expect(originalFollowUp).not.toHaveBeenCalled()
    queryState.linkedRuns = [
      {
        run_id: "run_msg",
        query: "Battery recycling supply chain",
        status: "waiting_human",
        phase: "awaiting_plan_review",
        control_state: "running",
        latest_checkpoint_id: "cp_1",
        updated_at: "2026-10-11T23:00:00+00:00"
      }
    ]
    view.rerender(
      <DemoModeProvider>
        <PlaygroundChat
          onAttachResearchContext={currentAttach}
          onPrepareResearchFollowUp={currentFollowUp}
        />
      </DemoModeProvider>
    )
    const row = within(screen.getByTestId("playground-message-mock"))
    expect(row.getByText("Plan review needed")).toBeInTheDocument()
    expect(row.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=run_msg"
    )
    expect(
      row.queryByRole("button", { name: "Use in Chat" })
    ).not.toBeInTheDocument()
    expect(
      row.queryByRole("button", { name: "Follow up" })
    ).not.toBeInTheDocument()
    expect(useMessageOptionState.value.messages[0].metadataExtra).toBe(
      message.metadataExtra
    )
  })

})
