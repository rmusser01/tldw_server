// @vitest-environment jsdom
/**
 * TASK-15623: an idle /chat tab must not re-render its message rows.
 *
 * PlaygroundMessage is memoized with a strict comparator, so every row prop
 * PlaygroundChat passes must keep its identity while the messages are
 * unchanged. useMessageOption re-creates its actions every render in
 * production; this mock does the same, so the row handlers must be stable
 * wrappers that still call the hook's LATEST action at call time, with the
 * same arguments and return value (the TASK-13521 side-panel pattern; the
 * TASK-13511 handlers that dropped them were removed by TASK-13570).
 */
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"

const harness = vi.hoisted(() => {
  const makeActions = () => ({
    editMessage: vi.fn(async () => true),
    deleteMessage: vi.fn(async () => true),
    toggleMessagePinned: vi.fn(async () => undefined),
    createChatBranch: vi.fn(async () => undefined),
    regenerateLastMessage: vi.fn(async () => undefined),
    onSubmit: vi.fn(async () => undefined),
    stopStreamingRequest: vi.fn()
  })
  return {
    makeActions,
    latest: makeActions(),
    chatRenders: 0,
    rowRenders: new Map<string, number>(),
    rowProps: new Map<string, Record<string, any>>(),
    changedRowProps: new Set<string>(),
    researchQueryOptions: null as Record<string, any> | null,
    state: {} as Record<string, any>
  }
})

const clientMocks = vi.hoisted(() => ({
  initialize: vi.fn(async () => undefined),
  listChatResearchRuns: vi.fn(),
  getProvidersStatus: vi.fn(async () => ({})),
  getResearchBundle: vi.fn(async () => ({}))
}))

const audioStatusMock = vi.hoisted(() =>
  // Production returns a fresh object every call.
  vi.fn(() => ({ healthState: "healthy", voicesAvailable: true }))
)

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => {
    harness.chatRenders += 1
    harness.latest = harness.makeActions()
    return { ...harness.state, ...harness.latest }
  }
}))

vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const idle = { data: undefined, isSuccess: false, isError: false, refetch: async () => undefined }
  return {
    ...actual,
    useQuery: (options: Record<string, any>) => {
      if (options?.queryKey?.[0] === "playground:chat-linked-research-runs") {
        harness.researchQueryOptions = options
        return actual.useQuery(options as any)
      }
      return idle
    }
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: clientMocks }))

vi.mock("@/hooks/useTldwAudioStatus", () => ({ useTldwAudioStatus: audioStatusMock }))

vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<typeof import("@/hooks/chat/useHistorySelection")>()),
  useHistorySelectionContext: () => null
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue]
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => null
}))

vi.mock("../PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => null
}))

// Memoized row spy: default shallow equality, which is the real comparator's
// contract for every prop this test exercises.
vi.mock("@/components/Common/Playground/Message", () => {
  const Row = (props: Record<string, any>) => {
    const id = String(props.messageId)
    harness.rowProps.set(id, props)
    harness.rowRenders.set(id, (harness.rowRenders.get(id) ?? 0) + 1)
    return <div data-testid="row">{props.message}</div>
  }
  return {
    PlaygroundMessage: React.memo(Row, (prev, next) => {
      const changed = Object.keys(next).filter((key) => !Object.is(prev[key], next[key]))
      changed.forEach((key) => harness.changedRowProps.add(key))
      harness.rowProps.set(String(next.messageId), next)
      return changed.length === 0 && Object.keys(prev).length === Object.keys(next).length
    })
  }
})

const RESEARCH_KEY = ["playground:chat-linked-research-runs", "chat-1"]
const completedRun = () => ({
  run_id: "rs_done",
  query: "Done query",
  status: "completed",
  phase: "completed",
  control_state: "running",
  latest_checkpoint_id: null,
  updated_at: "2026-03-08T20:03:00+00:00"
})

const buildMessages = (count: number) =>
  Array.from({ length: count }, (_unused, index) => ({
    id: `m${index}`,
    isBot: index % 2 === 1,
    role: index % 2 === 1 ? "assistant" : "user",
    name: index % 2 === 1 ? "Assistant" : "You",
    message: `message ${index}`
  }))

const settle = () => new Promise((resolve) => setTimeout(resolve, 20))

let queryClient: QueryClient
const tree = () => (
  <QueryClientProvider client={queryClient}>
    <PlaygroundChat />
  </QueryClientProvider>
)
const totalRowRenders = () => Array.from(harness.rowRenders.values()).reduce((sum, count) => sum + count, 0)

beforeEach(() => {
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  harness.chatRenders = 0
  harness.rowRenders.clear()
  harness.rowProps.clear()
  harness.changedRowProps.clear()
  harness.researchQueryOptions = null
  audioStatusMock.mockClear()
  harness.state = {
    messages: buildMessages(6),
    setMessages: vi.fn(),
    streaming: false,
    isProcessing: false,
    isSearchingInternet: false,
    ttsEnabled: false,
    actionInfo: null,
    messageSteeringMode: "none",
    setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false,
    setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(),
    createCompareBranch: vi.fn(),
    temporaryChat: false,
    serverChatId: null,
    serverChatCharacterId: null,
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
    compareMaxModels: 3,
    selectedAssistant: null
  }
})

describe("PlaygroundChat row memoization (TASK-15623)", () => {
  it("re-rendering with unchanged messages re-renders no message row", () => {
    const { rerender } = render(tree())
    expect(totalRowRenders()).toBe(6)
    harness.rowRenders.clear()
    const before = harness.chatRenders

    rerender(tree())

    expect(harness.chatRenders).toBe(before + 1)
    expect(Array.from(harness.changedRowProps)).toEqual([])
    expect(totalRowRenders()).toBe(0)
  })

  it("passes every row the same audio status from one hook call per render", () => {
    render(tree())
    const statuses = Array.from(harness.rowProps.values()).map((props) => props.audioStatus)
    expect(statuses[0]).toEqual({ healthState: "healthy", voicesAvailable: true })
    expect(new Set(statuses).size).toBe(1)
    expect(audioStatusMock).toHaveBeenCalledTimes(harness.chatRenders)
  })

  it("the row edit handler forwards all four arguments to the latest editMessage and returns its promise", () => {
    const { rerender } = render(tree())
    const handler = harness.rowProps.get("m1")!.onEditFormSubmit
    const firstEdit = harness.latest.editMessage

    rerender(tree())
    expect(harness.rowProps.get("m1")!.onEditFormSubmit).toBe(handler)

    const isCurrent = () => true
    const isOriginCurrent = () => true
    const outcome = handler("edited", true, isCurrent, isOriginCurrent)

    expect(firstEdit).not.toHaveBeenCalled()
    expect(harness.latest.editMessage).toHaveBeenCalledWith(1, "edited", false, true, isCurrent, isOriginCurrent)
    expect(outcome).toBe(harness.latest.editMessage.mock.results[0].value)
  })

  it("a row's edit handler follows the row to its new index", () => {
    const { rerender } = render(tree())
    harness.state.messages = harness.state.messages.slice(1)

    rerender(tree())
    harness.rowProps.get("m3")!.onEditFormSubmit("edited", false)

    expect(harness.latest.editMessage).toHaveBeenCalledWith(2, "edited", false, false, undefined, undefined)
  })

  it("the delete source resolves the latest deleteMessage through .current and passes its result through", () => {
    const { rerender } = render(tree())
    const source = harness.rowProps.get("m2")!.onDeleteMessage
    rerender(tree())
    expect(harness.rowProps.get("m2")!.onDeleteMessage).toBe(source)
    expect(typeof source).toBe("function")
    expect("current" in source).toBe(true)

    // A confirmation resolves `.current` when it opens and keeps that action.
    const atOpen = harness.latest.deleteMessage
    const confirmed = source.current
    rerender(tree())
    const accepted = confirmed()
    expect(atOpen).toHaveBeenCalledWith(2)
    expect(harness.latest.deleteMessage).not.toHaveBeenCalled()
    expect(accepted).toBe(atOpen.mock.results[0].value)

    // A direct call dispatches to the latest action.
    const direct = source()
    expect(harness.latest.deleteMessage).toHaveBeenCalledWith(2)
    expect(direct).toBe(harness.latest.deleteMessage.mock.results[0].value)
  })

  it("the other row actions call the hook's latest actions for the row they were rendered for", () => {
    const { rerender } = render(tree())
    const props = harness.rowProps.get("m3")!
    const first = harness.latest
    rerender(tree())

    props.onTogglePinned()
    props.onNewBranch()
    const regenerated = props.onRegenerate("dispatch")
    props.onContinue()

    expect(harness.latest.toggleMessagePinned).toHaveBeenCalledWith(3)
    expect(harness.latest.createChatBranch).toHaveBeenCalledWith("m3")
    expect(harness.latest.regenerateLastMessage).toHaveBeenCalledWith("dispatch")
    expect(regenerated).toBe(harness.latest.regenerateLastMessage.mock.results[0].value)
    expect(harness.latest.onSubmit).toHaveBeenCalledWith({ image: "", message: "", isContinue: true })
    for (const action of Object.values(first)) expect(action).not.toHaveBeenCalled()
  })
})

describe("PlaygroundChat linked research poll (TASK-15623)", () => {
  beforeEach(() => {
    harness.state.serverChatId = "chat-1"
    // A fresh, deep-equal payload per fetch, as the server returns it.
    clientMocks.listChatResearchRuns.mockReset()
    clientMocks.listChatResearchRuns.mockImplementation(async () => ({ runs: [completedRun()] }))
  })

  const refetch = () =>
    act(async () => {
      await queryClient.refetchQueries({ queryKey: RESEARCH_KEY })
      await settle()
    })

  const waitForFirstFetch = async () => {
    await waitFor(() => expect(queryClient.getQueryState(RESEARCH_KEY)?.status).toBe("success"))
    await act(settle)
  }

  it("a refetch that returns identical runs does not re-render PlaygroundChat", async () => {
    render(tree())
    await waitForFirstFetch()
    const before = harness.chatRenders

    await refetch()

    expect(clientMocks.listChatResearchRuns).toHaveBeenCalledTimes(2)
    expect(harness.chatRenders).toBe(before)
  })

  it("three consecutive failures slow the poll to 60 s and a success restores it", async () => {
    render(tree())
    await waitForFirstFetch()
    const interval = () =>
      harness.researchQueryOptions!.refetchInterval(queryClient.getQueryCache().find({ queryKey: RESEARCH_KEY }))
    expect(interval()).toBe(30_000)

    clientMocks.listChatResearchRuns.mockImplementation(async () => {
      throw new Error("research unavailable")
    })
    for (const expected of [30_000, 30_000, 60_000]) {
      await refetch()
      expect(queryClient.getQueryState(RESEARCH_KEY)?.status).toBe("error")
      expect(interval()).toBe(expected)
    }

    clientMocks.listChatResearchRuns.mockImplementation(async () => ({ runs: [completedRun()] }))
    await refetch()
    expect(interval()).toBe(30_000)
  })
})
