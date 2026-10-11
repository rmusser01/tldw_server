import React from "react"
import { App, ConfigProvider } from "antd"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { PlaygroundChat } from "../PlaygroundChat"
import { PlaygroundMessage } from "@/components/Common/Playground/Message"
import type { ChatLinkedResearchRun } from "@/services/tldw/TldwApiClient"

const state = vi.hoisted(() => ({
  runs: [] as ChatLinkedResearchRun[],
  realQueries: false,
  selection: null as {
    view: { owner_key: string; conversation_id: string }
    owner: { kind: "native"; conversation_id: string; validate_lease: () => boolean }
  } | null,
  messageOptions: {
    messages: [] as (typeof completionMessage)[], setMessages: vi.fn(), setHistory: vi.fn(),
    streaming: false, isProcessing: false, isSearchingInternet: false,
    regenerateLastMessage: vi.fn(), editMessage: vi.fn(), deleteMessage: vi.fn(),
    toggleMessagePinned: vi.fn(), ttsEnabled: false, onSubmit: vi.fn(), actionInfo: null,
    messageSteeringMode: "none", setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false, setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(), createChatBranch: vi.fn(), createCompareBranch: vi.fn(),
    temporaryChat: false, serverChatId: "chat-1", serverChatCharacterId: null,
    stopStreamingRequest: vi.fn(), isEmbedding: false, compareMode: false,
    compareFeatureEnabled: false, compareSelectionByCluster: {},
    setCompareSelectionForCluster: vi.fn(), compareActiveModelsByCluster: {},
    setCompareActiveModelsForCluster: vi.fn(), setCompareSelectedModels: vi.fn(),
    historyId: "history-1", setSelectedModel: vi.fn(), setCompareMode: vi.fn(),
    sendPerModelReply: vi.fn(), compareCanonicalByCluster: {},
    setCompareCanonicalForCluster: vi.fn(), compareContinuationModeByCluster: {},
    setCompareContinuationModeForCluster: vi.fn(), setCompareParentForHistory: vi.fn(),
    compareSplitChats: {}, setCompareSplitChat: vi.fn(), compareMaxModels: 3
  }
}))

const clients = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue(undefined),
  listChatResearchRuns: vi.fn(),
  getResearchBundle: vi.fn().mockResolvedValue({
    question: "Battery recycling", outline: { sections: [{ title: "Overview" }] },
    claims: [{ text: "Claim one" }], unresolved_questions: [],
    verification_summary: { unsupported_claim_count: 0 }, source_trust: []
  })
}))

vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => state.messageOptions }))
vi.mock("@/hooks/chat/useHistorySelection", () => ({ useHistorySelectionContext: () => {
  const selection = state.selection
  return selection ? {
    ...selection,
    getCurrent: () => state.selection,
    fence: () => () => state.selection === selection
  } : null
} }))
vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>("@tanstack/react-query")
  return {
    ...actual,
    useQuery: (options: Parameters<typeof actual.useQuery>[0]) => state.realQueries
      ? actual.useQuery(options)
      : {
          data: options.queryKey[0] === "playground:chat-linked-research-runs" ? { runs: state.runs } : [],
          isFetched: true, isSuccess: true, isError: false, dataUpdatedAt: 1, errorUpdatedAt: 0,
          refetch: vi.fn()
        }
  }
})
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()] }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key }) }))
vi.mock("@/hooks/useConnectionState", () => ({ useIsConnected: () => false }))
vi.mock("@/hooks/useAntdNotification", () => ({ useAntdNotification: () => ({ success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() }) }))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ cancel: vi.fn(), speak: vi.fn(), isSpeaking: false }) }))
vi.mock("@/hooks/useFeedback", () => ({ useFeedback: () => ({ thumb: null, detail: "", sourceFeedback: {}, canSubmit: false, isSubmitting: false, showThanks: false }) }))
vi.mock("@/hooks/useImplicitFeedback", () => ({ useImplicitFeedback: () => ({ trackDwellTime: vi.fn() }) }))
vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => ({ capabilities: {} }) }))
vi.mock("@/hooks/useTldwAudioStatus", () => ({ useTldwAudioStatus: () => ({ healthState: "ready", voicesAvailable: false }) }))
vi.mock("@/hooks/useDiscoSkills", () => ({ useDiscoSkills: () => ({ enabled: false, stats: null, triggerProbabilityBase: 0, persistComments: false }) }))
vi.mock("@/components/Common/Playground/MessageActionsBar", () => ({ MessageActionsBar: () => null }))
vi.mock("@/components/Sidepanel/Chat/FeedbackModal", () => ({ FeedbackModal: () => null }))
vi.mock("../PlaygroundEmpty", () => ({ PlaygroundEmpty: () => null }))
vi.mock("@/components/Common/ChatGreetingPicker", () => ({ ChatGreetingPicker: () => null }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: clients }))

const completionMessage = {
  id: "completion-1", isBot: true, role: "assistant", name: "Assistant",
  message: "Research finished.", sources: [],
  metadataExtra: { deep_research_completion: {
    run_id: "run-1", query: "Battery recycling", kind: "completion_handoff"
  } }
}
const reviewRun = {
  run_id: "run-1", query: "Battery recycling", status: "waiting_human",
  phase: "awaiting_plan_review", control_state: "running",
  latest_checkpoint_id: "checkpoint-1", updated_at: "2026-10-09T20:00:00Z"
}
const completedRun = { ...reviewRun, status: "completed", phase: "completed", latest_checkpoint_id: null }
const bundle = {
  question: "Battery recycling", outline: { sections: [{ title: "Overview" }] },
  claims: [{ text: "Claim one" }], unresolved_questions: [],
  verification_summary: { unsupported_claim_count: 0 }, source_trust: []
}
const held = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const frame = (children: React.ReactNode) => (
  <ConfigProvider theme={{ token: { motion: false } }}><App>{children}</App></ConfigProvider>
)
const messageControls = () => within(screen.getByTestId("chat-message"))

describe("research action freshness with real memoized message controls", () => {
  beforeEach(() => {
    state.runs = []
    state.realQueries = false
    state.selection = null
    state.messageOptions.serverChatId = "chat-1"
    state.messageOptions.temporaryChat = false
    state.messageOptions.messages = [completionMessage]
    clients.listChatResearchRuns.mockReset()
    clients.initialize.mockReset().mockResolvedValue(undefined)
    clients.getResearchBundle.mockReset().mockResolvedValue(bundle)
    vi.spyOn(globalThis, "fetch").mockImplementation(() => {
      throw new Error("Unexpected network request in finite research action test")
    })
  })
  afterEach(() => {
    expect(globalThis.fetch).not.toHaveBeenCalled()
    vi.restoreAllMocks()
  })

  it("replaces fallback actions with review controls when unchanged metadata gets a waiting_human policy", () => {
    const attach = vi.fn()
    const followUp = vi.fn()
    const tree = () => frame(<PlaygroundChat onAttachResearchContext={attach} onPrepareResearchFollowUp={followUp} />)
    const { rerender } = render(tree())
    expect(messageControls().getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
    expect(messageControls().getByRole("button", { name: "Follow up" })).toBeInTheDocument()

    state.runs = [reviewRun]
    rerender(tree())

    expect(messageControls().queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
    expect(messageControls().queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
    expect(messageControls().getByText("Plan review needed")).toBeInTheDocument()
    expect(messageControls().getByRole("link", { name: "Review in Research" })).toHaveAttribute("href", "/research?run=run-1")
  })

  it("replaces review controls with completion actions without replacing message metadata", () => {
    state.runs = [reviewRun]
    const attach = vi.fn()
    const followUp = vi.fn()
    const tree = () => frame(<PlaygroundChat onAttachResearchContext={attach} onPrepareResearchFollowUp={followUp} />)
    const { rerender } = render(tree())
    expect(messageControls().getByRole("link", { name: "Review in Research" })).toBeInTheDocument()

    state.runs = [completedRun]
    rerender(tree())

    expect(messageControls().queryByRole("link", { name: "Review in Research" })).not.toBeInTheDocument()
    expect(messageControls().queryByText("Plan review needed")).not.toBeInTheDocument()
    expect(messageControls().getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
    expect(messageControls().getByRole("button", { name: "Follow up" })).toBeInTheDocument()
  })

  it("retains same-chat review policy on a failed background refetch and applies later success", async () => {
    state.realQueries = true
    clients.listChatResearchRuns
      .mockResolvedValueOnce({ runs: [reviewRun] })
      .mockRejectedValueOnce(new Error("Finite refetch failure"))
      .mockResolvedValueOnce({ runs: [completedRun] })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
    const view = render(frame(<QueryClientProvider client={client}>
      <PlaygroundChat onAttachResearchContext={vi.fn()} onPrepareResearchFollowUp={vi.fn()} />
    </QueryClientProvider>))
    try {
      await waitFor(() => expect(messageControls().getByRole("link", { name: "Review in Research" })).toBeInTheDocument())
      const query = client.getQueryCache().findAll({ queryKey: ["playground:chat-linked-research-runs"] })[0]
      const retainedData = query.state.data

      await act(async () => {
        await client.refetchQueries({ queryKey: query.queryKey })
        await new Promise(resolve => setTimeout(resolve, 0))
      })
      expect(query.state.status).toBe("error")
      expect(query.state.data).toBe(retainedData)
      expect(messageControls().getByText("Plan review needed")).toBeInTheDocument()
      expect(messageControls().getByRole("link", { name: "Review in Research" })).toHaveAttribute("href", "/research?run=run-1")
      expect(messageControls().queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
      expect(messageControls().queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()

      await act(async () => {
        await client.refetchQueries({ queryKey: query.queryKey })
        await new Promise(resolve => setTimeout(resolve, 0))
      })
      expect(query.state.status).toBe("success")
      expect(messageControls().queryByRole("link", { name: "Review in Research" })).not.toBeInTheDocument()
      expect(messageControls().getByRole("button", { name: "Use in Chat" })).toBeInTheDocument()
      expect(messageControls().getByRole("button", { name: "Follow up" })).toBeInTheDocument()
    } finally {
      view.unmount()
      client.clear()
    }
  })

  it.each(["chat", "owner", "revoked owner"] as const)("does not reuse a prior policy across a %s boundary", async boundary => {
    state.realQueries = true
    state.selection = {
      view: { owner_key: "owner-a", conversation_id: "chat-1" },
      owner: { kind: "native", conversation_id: "chat-1", validate_lease: () => true }
    }
    clients.listChatResearchRuns.mockResolvedValueOnce({ runs: [reviewRun] })
      .mockRejectedValue(new Error("New scope has no loaded policy"))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
    const attach = vi.fn()
    const followUp = vi.fn()
    const tree = () => frame(<QueryClientProvider client={client}>
      <PlaygroundChat onAttachResearchContext={attach} onPrepareResearchFollowUp={followUp} />
    </QueryClientProvider>)
    const view = render(tree())
    try {
      await waitFor(() => expect(messageControls().getByText("Plan review needed")).toBeInTheDocument())
      if (boundary === "chat") {
        state.messageOptions.serverChatId = "chat-2"
        state.selection = { ...state.selection, view: { owner_key: "owner-a", conversation_id: "chat-2" },
          owner: { ...state.selection.owner, conversation_id: "chat-2" } }
      } else if (boundary === "owner") {
        state.selection = { ...state.selection, view: { owner_key: "owner-b", conversation_id: "chat-1" } }
      } else {
        state.selection = { ...state.selection, owner: { ...state.selection.owner, validate_lease: () => false } }
      }
      view.rerender(tree())

      await waitFor(() => expect(messageControls().queryByRole("link", { name: "Review in Research" })).not.toBeInTheDocument())
      expect(messageControls().queryByText("Plan review needed")).not.toBeInTheDocument()
      if (boundary === "revoked owner") {
        expect(messageControls().queryByRole("button", { name: "Use in Chat" })).not.toBeInTheDocument()
        expect(messageControls().queryByRole("button", { name: "Follow up" })).not.toBeInTheDocument()
        expect(attach).not.toHaveBeenCalled()
        expect(followUp).not.toHaveBeenCalled()
      }
      if (boundary !== "revoked owner") {
        await waitFor(() => expect(client.getQueryCache().findAll({ queryKey: ["playground:chat-linked-research-runs"] }).some(query => query.state.status === "error" && query.state.data === undefined)).toBe(true))
      }
    } finally {
      view.unmount()
      client.clear()
    }
  })

  it.each(["Use in Chat", "Follow up"])("rechecks a revoked lease when dispatching the previously rendered %s action", async label => {
    let valid = true
    state.selection = {
      view: { owner_key: "owner-a", conversation_id: "chat-1" },
      owner: { kind: "native", conversation_id: "chat-1", validate_lease: () => valid }
    }
    state.runs = [completedRun]
    const attach = vi.fn()
    const followUp = vi.fn()
    render(frame(<PlaygroundChat onAttachResearchContext={attach} onPrepareResearchFollowUp={followUp} />))
    const button = messageControls().getByRole("button", { name: label })
    valid = false
    await act(async () => { fireEvent.click(button) })
    expect(attach).not.toHaveBeenCalled()
    expect(followUp).not.toHaveBeenCalled()
    expect(clients.initialize).not.toHaveBeenCalled()
    expect(clients.getResearchBundle).not.toHaveBeenCalled()
  })

  it.each([
    ["chat", "initialize"], ["chat", "bundle"],
    ["owner", "initialize"], ["owner", "bundle"],
    ["revocation", "initialize"], ["revocation", "bundle"]
  ] as const)("does not attach research after %s while %s is held", async (boundary, phase) => {
    let valid = true
    state.selection = {
      view: { owner_key: "owner-a", conversation_id: "chat-1" },
      owner: { kind: "native", conversation_id: "chat-1", validate_lease: () => valid }
    }
    state.runs = [completedRun]
    const pending = held<unknown>()
    const method = phase === "initialize" ? clients.initialize : clients.getResearchBundle
    method.mockImplementationOnce(() => pending.promise)
    const attach = vi.fn()
    const tree = () => frame(<PlaygroundChat onAttachResearchContext={attach} onPrepareResearchFollowUp={vi.fn()} />)
    const view = render(tree())
    fireEvent.click(messageControls().getByRole("button", { name: "Use in Chat" }))
    await waitFor(() => expect(method).toHaveBeenCalledTimes(1))
    if (boundary === "chat") {
      state.messageOptions.serverChatId = "chat-2"
      state.selection = { view: { owner_key: "owner-a", conversation_id: "chat-2" },
        owner: { kind: "native", conversation_id: "chat-2", validate_lease: () => true } }
    } else if (boundary === "owner") {
      state.selection = { ...state.selection, view: { owner_key: "owner-b", conversation_id: "chat-1" },
        owner: { ...state.selection.owner, validate_lease: () => true } }
    } else {
      valid = false
    }
    view.rerender(tree())
    await act(async () => { pending.resolve(phase === "bundle" ? bundle : undefined); await pending.promise })
    expect(attach).not.toHaveBeenCalled()
    if (phase === "initialize") expect(clients.getResearchBundle).not.toHaveBeenCalled()
  })

  it.each(["initialize", "bundle"] as const)("keeps same-owner Use in Chat working while %s is held", async phase => {
    state.selection = {
      view: { owner_key: "owner-a", conversation_id: "chat-1" },
      owner: { kind: "native", conversation_id: "chat-1", validate_lease: () => true }
    }
    state.runs = [completedRun]
    const pending = held<unknown>()
    const method = phase === "initialize" ? clients.initialize : clients.getResearchBundle
    method.mockImplementationOnce(() => pending.promise)
    const attach = vi.fn()
    render(frame(<PlaygroundChat onAttachResearchContext={attach} />))
    fireEvent.click(messageControls().getByRole("button", { name: "Use in Chat" }))
    await waitFor(() => expect(method).toHaveBeenCalledTimes(1))
    await act(async () => { pending.resolve(phase === "bundle" ? bundle : undefined); await pending.promise })
    expect(attach).toHaveBeenCalledWith(expect.objectContaining({ run_id: "run-1" }))
  })

  it.each(["Use in Chat", "Follow up"])("invokes the updated PlaygroundChat %s callback with unchanged metadata", async label => {
    const Harness = ({ version }: { version: string }) => {
      const [result, setResult] = React.useState("")
      const attach = React.useCallback(() => setResult(`${version}: attached`), [version])
      const followUp = React.useCallback(() => setResult(`${version}: follow-up`), [version])
      return <><PlaygroundChat onAttachResearchContext={attach} onPrepareResearchFollowUp={followUp} /><output aria-label="Action result">{result}</output></>
    }
    const { rerender } = render(frame(<Harness version="first" />))
    fireEvent.click(messageControls().getByRole("button", { name: label }))
    await waitFor(() => expect(screen.getByLabelText("Action result")).toHaveTextContent(`first: ${label === "Use in Chat" ? "attached" : "follow-up"}`))

    rerender(frame(<Harness version="second" />))
    fireEvent.click(messageControls().getByRole("button", { name: label }))
    await waitFor(() => expect(screen.getByLabelText("Action result")).toHaveTextContent(`second: ${label === "Use in Chat" ? "attached" : "follow-up"}`))
  })

  it.each([
    ["onUseInChat", "Use in Chat"], ["onFollowUp", "Follow up"]
  ] as const)("updates a memoized message when only %s identity changes", (action, label) => {
    const stableProps: React.ComponentProps<typeof PlaygroundMessage> = {
      message: "Research finished.", isBot: true, role: "assistant", name: "Assistant",
      currentMessageIndex: 0, totalMessages: 1, onRegenerate: vi.fn(), onContinue: vi.fn(),
      onEditFormSubmit: vi.fn(), isProcessing: false, isStreaming: false,
      conversationInstanceId: "callback-test"
    }
    const Harness = ({ version }: { version: string }) => {
      const [result, setResult] = React.useState("")
      const callback = React.useCallback(() => setResult(version), [version])
      return <><PlaygroundMessage {...stableProps} researchActions={{ [action]: callback }} /><output aria-label="Action result">{result}</output></>
    }
    const { rerender } = render(frame(<Harness version="first" />))
    fireEvent.click(messageControls().getByRole("button", { name: label }))
    expect(screen.getByLabelText("Action result")).toHaveTextContent("first")

    rerender(frame(<Harness version="second" />))
    fireEvent.click(messageControls().getByRole("button", { name: label }))
    expect(screen.getByLabelText("Action result")).toHaveTextContent("second")
  })
})
