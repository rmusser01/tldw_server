// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"
import { DemoModeProvider } from "@/context/demo-mode"

type MockLinkedRun = {
  run_id: string
  query: string
  status: string
  phase: string
  control_state: string
  latest_checkpoint_id: string | null
  updated_at: string
}

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
  linkedRuns: [] as MockLinkedRun[],
  linkedRunsStatus: "success" as "success" | "error",
  errorUpdatedAt: 0,
  dataUpdatedAt: 1,
  capturedOptions: null as Record<string, any> | null
}))

const notificationMocks = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn()
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

const mockComponents = vi.hoisted(() => ({
  PlaygroundEmpty: () => <div data-testid="playground-empty" />
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
        queryState.capturedOptions = options
        return {
          data: { runs: queryState.linkedRuns },
          isSuccess: queryState.linkedRunsStatus === "success",
          isError: queryState.linkedRunsStatus === "error",
          status: queryState.linkedRunsStatus,
          errorUpdatedAt: queryState.errorUpdatedAt,
          dataUpdatedAt: queryState.dataUpdatedAt
        }
      }
      return { data: [], isSuccess: true, isError: false, status: "success", errorUpdatedAt: 0, dataUpdatedAt: 1 }
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
  useAntdNotification: () => notificationMocks
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: clientMocks
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => <div data-testid="chat-greeting-picker" />
}))

vi.mock("./PlaygroundEmpty", () => ({
  PlaygroundEmpty: mockComponents.PlaygroundEmpty
}))

vi.mock("../PlaygroundEmpty", () => ({
  PlaygroundEmpty: mockComponents.PlaygroundEmpty
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: { message: string }) => (
    <div data-testid="playground-message-mock">{props.message}</div>
  )
}))

const setLinkedRuns = (runs: MockLinkedRun[]) => {
  queryState.linkedRuns = runs
  queryState.linkedRunsStatus = "success"
  queryState.dataUpdatedAt += 1
}

const setLinkedRunError = () => {
  queryState.linkedRunsStatus = "error"
  queryState.errorUpdatedAt += 1
}

const renderChat = (extraProps?: Record<string, unknown>) =>
  render(
    <DemoModeProvider>
      <PlaygroundChat {...(extraProps as any)} />
    </DemoModeProvider>
  )

const renderChatTree = () => (
  <DemoModeProvider>
    <PlaygroundChat />
  </DemoModeProvider>
)

describe("PlaygroundChat linked research status integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useMessageOptionState.value.messages = []
    useMessageOptionState.value.temporaryChat = false
    useMessageOptionState.value.serverChatId = "chat-1"
    queryState.linkedRuns = []
    queryState.linkedRunsStatus = "success"
    queryState.errorUpdatedAt = 0
    queryState.dataUpdatedAt = 1
    queryState.capturedOptions = null
  })

  it("renders the linked research status block below empty scaffolding and above the transcript area", async () => {
    setLinkedRuns([
      {
        run_id: "rs_wait",
        query: "Check original source",
        status: "waiting_human",
        phase: "awaiting_plan_review",
        control_state: "running",
        latest_checkpoint_id: "cp_1",
        updated_at: "2026-03-08T20:00:00+00:00"
      }
    ])

    renderChat()

    const emptyState = screen.getByTestId("playground-empty")
    const statusBlock = await screen.findByTestId("research-run-status-stack")
    expect(statusBlock).toBeInTheDocument()
    expect(emptyState.compareDocumentPosition(statusBlock)).toBe(Node.DOCUMENT_POSITION_FOLLOWING)
    expect(screen.getByText("Needs review")).toBeInTheDocument()
    expect(screen.getByText("Plan review needed")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=rs_wait"
    )
  })

  it("renders multiple linked runs with distinct labels and stacked rows", async () => {
    setLinkedRuns([
      {
        run_id: "rs_running",
        query: "Running query",
        status: "running",
        phase: "collecting",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      },
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:02:00+00:00"
      },
      {
        run_id: "rs_failed",
        query: "Failed query",
        status: "failed",
        phase: "failed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:01:00+00:00"
      }
    ])

    renderChat()

    expect(await screen.findAllByTestId("research-run-status-row")).toHaveLength(3)
    expect(await screen.findByText("Running")).toBeInTheDocument()
    expect(await screen.findByText("Completed")).toBeInTheDocument()
    expect(await screen.findByText("Failed")).toBeInTheDocument()
  })

  it("shows Use in Chat only for completed linked runs", async () => {
    setLinkedRuns([
      {
        run_id: "rs_running",
        query: "Running query",
        status: "running",
        phase: "collecting",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      },
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:02:00+00:00"
      }
    ])

    renderChat()

    const rows = await screen.findAllByTestId("research-run-status-row")
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(within(rows[1]).getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
  })

  it("clicking Use in Chat on a completed linked run fetches the bundle and attaches bounded context", async () => {
    setLinkedRuns([
      {
        run_id: "rs_completed",
        query: "Battery recycling supply chain",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:02:00+00:00"
      }
    ])

    const attachSpy = vi.fn()
    render(
      <DemoModeProvider>
        <PlaygroundChat onAttachResearchContext={attachSpy} />
      </DemoModeProvider>
    )

    fireEvent.click(await screen.findByRole("button", { name: "Use in Chat" }))

    await waitFor(() =>
      expect(clientMocks.getResearchBundle).toHaveBeenCalledWith("rs_completed")
    )
    await waitFor(() =>
      expect(attachSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          run_id: "rs_completed",
          query: "Battery recycling supply chain",
          question: "What changed in the battery recycling market?",
          research_url: "/research?run=rs_completed"
        })
      )
    )
  })

  it("shows Follow up only on completed linked runs and routes it through the shared preparation path", async () => {
    setLinkedRuns([
      {
        run_id: "rs_running",
        query: "Running query",
        status: "running",
        phase: "collecting",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      },
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:02:00+00:00"
      },
      {
        run_id: "rs_failed",
        query: "Failed query",
        status: "failed",
        phase: "failed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:01:00+00:00"
      }
    ])

    const followUpSpy = vi.fn()
    renderChat({ onPrepareResearchFollowUp: followUpSpy })

    const rows = await screen.findAllByTestId("research-run-status-row")
    expect(rows).toHaveLength(3)
    expect(within(rows[0]).queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
    expect(within(rows[1]).getByRole("button", { name: "Follow up" })).toBeInTheDocument()
    expect(within(rows[2]).queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()

    fireEvent.click(within(rows[1]).getByRole("button", { name: "Follow up" }))

    await waitFor(() =>
      expect(followUpSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          run_id: "rs_completed",
          query: "Completed query"
        })
      )
    )
  })

  it("shows a returned-from-research banner for a completed run with completed actions", () => {
    const dismissSpy = vi.fn()

    setLinkedRuns([
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:02:00+00:00"
      }
    ])

    renderChat({
      returnedResearchRunId: "rs_completed",
      onDismissReturnedResearchRun: dismissSpy,
      onPrepareResearchFollowUp: vi.fn(),
      onAttachResearchContext: vi.fn()
    })

    const banner = screen.getByTestId("returned-research-banner")
    expect(within(banner).getByText("Returned from Research")).toBeInTheDocument()
    expect(within(banner).getByText("Completed query")).toBeInTheDocument()
    expect(within(banner).getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
    expect(within(banner).getByRole("button", { name: "Follow up" })).toBeInTheDocument()
    expect(within(banner).getByRole("link", { name: "Open in Research" })).toHaveAttribute(
      "href",
      "/research?run=rs_completed"
    )

    fireEvent.click(within(banner).getByRole("button", { name: "Dismiss" }))
    expect(dismissSpy).toHaveBeenCalledTimes(1)
  })

  it("shows review-only returned-from-research actions when the returned run needs checkpoint review", () => {
    setLinkedRuns([
      {
        run_id: "rs_plan",
        query: "Plan review query",
        status: "waiting_human",
        phase: "awaiting_plan_review",
        control_state: "running",
        latest_checkpoint_id: "cp_plan",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ])

    renderChat({
      returnedResearchRunId: "rs_plan",
      onDismissReturnedResearchRun: vi.fn(),
      onPrepareResearchFollowUp: vi.fn(),
      onAttachResearchContext: vi.fn()
    })

    const banner = screen.getByTestId("returned-research-banner")
    expect(within(banner).getByText("Returned from Research")).toBeInTheDocument()
    expect(within(banner).getByText("Plan review needed")).toBeInTheDocument()
    expect(within(banner).getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=rs_plan"
    )
    expect(within(banner).queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(within(banner).queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
  })

  it("shows a review-cleared return banner with a stronger continue CTA for explicit returns that no longer need review", async () => {
    const attachSpy = vi.fn()

    setLinkedRuns([
      {
        run_id: "rs_reviewed",
        query: "Reviewed research query",
        status: "running",
        phase: "collecting",
        control_state: "running",
        latest_checkpoint_id: "cp_reviewed",
        updated_at: "2026-03-19T20:04:00+00:00"
      }
    ])

    renderChat({
      returnedResearchRunId: "rs_reviewed",
      onDismissReturnedResearchRun: vi.fn(),
      onPrepareResearchFollowUp: vi.fn(),
      onAttachResearchContext: attachSpy
    })

    const banner = screen.getByTestId("returned-research-banner")
    expect(within(banner).getByText("Returned from Research")).toBeInTheDocument()
    expect(
      within(banner).getByText("Your review is reflected in this run.")
    ).toBeInTheDocument()
    expect(
      within(banner).getByRole("button", {
        name: "Continue with reviewed research"
      })
    ).toBeInTheDocument()
    expect(within(banner).getByRole("link", { name: "Open in Research" })).toHaveAttribute(
      "href",
      "/research?run=rs_reviewed"
    )
    expect(within(banner).queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()

    fireEvent.click(
      within(banner).getByRole("button", {
        name: "Continue with reviewed research"
      })
    )

    await waitFor(() =>
      expect(clientMocks.getResearchBundle).toHaveBeenCalledWith("rs_reviewed")
    )
    await waitFor(() =>
      expect(attachSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          run_id: "rs_reviewed",
          query: "Reviewed research query"
        })
      )
    )
  })

  it("does not show a returned-from-research banner when the run is missing from linked runs", () => {
    setLinkedRuns([
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:02:00+00:00"
      }
    ])

    renderChat({
      returnedResearchRunId: "rs_missing",
      onDismissReturnedResearchRun: vi.fn(),
      onPrepareResearchFollowUp: vi.fn(),
      onAttachResearchContext: vi.fn()
    })

    expect(screen.queryByText("Returned from Research")).not.toBeInTheDocument()
  })

  it("renders plan review handoff rows with a specific reason label and review action", async () => {
    setLinkedRuns([
      {
        run_id: "rs_plan",
        query: "Plan review query",
        status: "waiting_human",
        phase: "awaiting_plan_review",
        control_state: "running",
        latest_checkpoint_id: "cp_plan",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ])

    renderChat({ onPrepareResearchFollowUp: vi.fn(), onAttachResearchContext: vi.fn() })

    const row = await screen.findByTestId("research-run-status-row")
    expect(within(row).getByText("Plan review needed")).toBeInTheDocument()
    expect(within(row).getByRole("link", { name: "Review in Research" })).toHaveAttribute(
      "href",
      "/research?run=rs_plan"
    )
    expect(within(row).queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(within(row).queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
  })

  it("renders sources review handoff labels for both supported source-review phases", async () => {
    setLinkedRuns([
      {
        run_id: "rs_sources_1",
        query: "Source review query",
        status: "waiting_human",
        phase: "awaiting_source_review",
        control_state: "running",
        latest_checkpoint_id: "cp_sources_1",
        updated_at: "2026-03-19T20:03:00+00:00"
      },
      {
        run_id: "rs_sources_2",
        query: "Sources review query",
        status: "waiting_human",
        phase: "awaiting_sources_review",
        control_state: "running",
        latest_checkpoint_id: "cp_sources_2",
        updated_at: "2026-03-19T20:02:00+00:00"
      }
    ])

    renderChat()

    const rows = await screen.findAllByTestId("research-run-status-row")
    expect(within(rows[0]).getByText("Sources review needed")).toBeInTheDocument()
    expect(within(rows[0]).getByRole("link", { name: "Review in Research" })).toBeInTheDocument()
    expect(within(rows[1]).getByText("Sources review needed")).toBeInTheDocument()
    expect(within(rows[1]).getByRole("link", { name: "Review in Research" })).toBeInTheDocument()
  })

  it("renders outline review handoff labels for checkpoint-needed rows", async () => {
    setLinkedRuns([
      {
        run_id: "rs_outline",
        query: "Outline review query",
        status: "waiting_human",
        phase: "awaiting_outline_review",
        control_state: "running",
        latest_checkpoint_id: "cp_outline",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ])

    renderChat()

    const row = await screen.findByTestId("research-run-status-row")
    expect(within(row).getByText("Outline review needed")).toBeInTheDocument()
    expect(within(row).getByRole("link", { name: "Review in Research" })).toBeInTheDocument()
  })

  it("falls back to a generic review-needed label for unknown waiting_human phases", async () => {
    setLinkedRuns([
      {
        run_id: "rs_unknown_review",
        query: "Unknown review query",
        status: "waiting_human",
        phase: "awaiting_custom_review",
        control_state: "running",
        latest_checkpoint_id: "cp_custom",
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ])

    renderChat()

    const row = await screen.findByTestId("research-run-status-row")
    expect(within(row).getByText("Review needed")).toBeInTheDocument()
    expect(within(row).getByRole("link", { name: "Review in Research" })).toBeInTheDocument()
  })

  it("keeps completed-run actions unchanged while checkpoint handoff rows are stricter", async () => {
    setLinkedRuns([
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-19T20:03:00+00:00"
      }
    ])

    renderChat({ onPrepareResearchFollowUp: vi.fn(), onAttachResearchContext: vi.fn() })

    const row = await screen.findByTestId("research-run-status-row")
    expect(within(row).getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
    expect(within(row).getByRole("button", { name: "Follow up" })).toBeInTheDocument()
    expect(within(row).getByRole("link", { name: "Open in Research" })).toHaveAttribute(
      "href",
      "/research?run=rs_completed"
    )
    expect(within(row).queryByText("Review needed")).not.toBeInTheDocument()
  })

  it("does not render the status block for temporary chats or chats without a server id", () => {
    setLinkedRuns([
      {
        run_id: "rs_running",
        query: "Running query",
        status: "running",
        phase: "collecting",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      }
    ])
    useMessageOptionState.value.temporaryChat = true

    const { rerender } = renderChat()
    expect(screen.queryByTestId("research-run-status-stack")).not.toBeInTheDocument()

    useMessageOptionState.value.temporaryChat = false
    useMessageOptionState.value.serverChatId = null
    rerender(
      <DemoModeProvider>
        <PlaygroundChat />
      </DemoModeProvider>
    )

    expect(screen.queryByTestId("research-run-status-stack")).not.toBeInTheDocument()
  })

  it("uses active polling for nonterminal runs, slows for terminal runs, backs off after repeated errors, and resets after success", async () => {
    setLinkedRuns([
      {
        run_id: "rs_running",
        query: "Running query",
        status: "running",
        phase: "collecting",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      }
    ])

    const { rerender } = renderChat()
    expect(
      queryState.capturedOptions?.refetchInterval?.({
        state: { data: { runs: queryState.linkedRuns } }
      })
    ).toBe(5_000)

    setLinkedRuns([
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      }
    ])
    rerender(renderChatTree())
    expect(
      queryState.capturedOptions?.refetchInterval?.({
        state: { data: { runs: queryState.linkedRuns } }
      })
    ).toBe(30_000)

    setLinkedRunError()
    rerender(renderChatTree())
    await waitFor(() =>
      expect(
        queryState.capturedOptions?.refetchInterval?.({
          state: { data: { runs: [{ status: "completed" }] } }
        })
      ).toBe(30_000)
    )
    setLinkedRunError()
    rerender(renderChatTree())
    await waitFor(() =>
      expect(
        queryState.capturedOptions?.refetchInterval?.({
          state: { data: { runs: [{ status: "completed" }] } }
        })
      ).toBe(30_000)
    )
    setLinkedRunError()
    rerender(renderChatTree())
    await waitFor(() =>
      expect(
        queryState.capturedOptions?.refetchInterval?.({
          state: { data: { runs: [{ status: "completed" }] } }
        })
      ).toBe(60_000)
    )

    setLinkedRuns([
      {
        run_id: "rs_completed",
        query: "Completed query",
        status: "completed",
        phase: "completed",
        control_state: "running",
        latest_checkpoint_id: null,
        updated_at: "2026-03-08T20:03:00+00:00"
      }
    ])
    rerender(renderChatTree())
    await waitFor(() =>
      expect(
        queryState.capturedOptions?.refetchInterval?.({
          state: { data: { runs: queryState.linkedRuns } }
        })
      ).toBe(30_000)
    )
  })

  it("keeps linked-run query failures silent and non-blocking", () => {
    setLinkedRunError()

    renderChat()

    expect(screen.queryByTestId("research-run-status-stack")).not.toBeInTheDocument()
    expect(screen.getByTestId("playground-empty")).toBeInTheDocument()
    expect(notificationMocks.error).not.toHaveBeenCalled()
    expect(notificationMocks.warning).not.toHaveBeenCalled()
  })
})
