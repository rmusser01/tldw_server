// @vitest-environment jsdom
import React from "react"
import { act, render } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { resetSavedReplyRefusals, rememberSavedReplyRefusal } from "@/hooks/chat/saved-reply-actions"
import type { NativeHistoryRegenerate } from "@/types/chat-modes"
import { SidePanelBody } from "../body"

// CM-03 (b): the side panel offers Continue on the last reply, and a saved
// reply's recorded status (Output limit reached, Interrupted) offers its
// actions, with the same rules as the full page.
const rowProps = vi.hoisted(() => new Map<string, Record<string, unknown>>())
const selection = vi.hoisted(() => ({ current: null as unknown }))
const sidepanelMessageState = vi.hoisted(() => ({
  value: {
    messages: [] as any[],
    setMessages: vi.fn(),
    streaming: false,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    continueLastMessage: vi.fn(async () => ({ status: "submitted" })),
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    isSearchingInternet: false,
    createChatBranch: vi.fn(),
    historyId: "history-1",
    temporaryChat: false,
    stopStreamingRequest: vi.fn(),
    serverChatId: "chat-1" as string | null,
    isEmbedding: false
  }
}))

vi.mock("~/hooks/useMessage", () => ({
  useMessage: () => sidepanelMessageState.value
}))

vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<typeof import("@/hooks/chat/useHistorySelection")>()),
  useHistorySelectionContext: () => selection.current
}))

vi.mock("~/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: Record<string, unknown> & { messageId?: string }) => {
    rowProps.set(props.messageId ?? "unknown", props)
    return <article data-testid="sidepanel-row" />
  }
}))

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 120,
    getVirtualItems: () =>
      Array.from({ length: count }, (_unused, index) => ({ index, key: `row-${index}`, start: index * 120 })),
    measureElement: vi.fn(),
    scrollToIndex: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => React.useState(defaultValue)
}))

vi.mock("@/store/webui", () => ({ useWebUI: () => ({ ttsEnabled: false }) }))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) => selector({ mode: "pro" })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({ useSelectedCharacter: () => [null] }))

vi.mock("@/store/option", () => {
  const optionState = {
    serverChatCharacterId: null,
    isLoading: false,
    streaming: false,
    isProcessing: false,
    chatDraftAddress: null,
    temporaryChat: false
  }
  return {
    useStoreMessageOption: Object.assign(
      (selector: (state: Record<string, unknown>) => unknown) => selector(optionState),
      { getState: () => optionState, subscribe: () => () => {} }
    )
  }
})

vi.mock("../empty", () => ({ EmptySidePanel: () => <div data-testid="empty-sidepanel" /> }))

vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "generated-id" }))

const view = {
  view_session_id: "view",
  owner_key: "owner",
  conversation_id: "chat-1",
  interpretation: { kind: "parent_graph_v1" as const },
  cursor: { kind: "after_message" as const, message_id: "a1" },
  selection_revision: 1
}

const node = (id: string, role: string, parent_id: string | null) => ({
  id,
  revision: "1",
  role,
  parent_id,
  settled: true
})

const savedChat = (owner: { kind: string } = { kind: "native" }) => ({
  getCurrent: () => ({
    status: "ready",
    owner,
    view,
    capture: {
      status: "captured",
      snapshot: { nodes: [] },
      rows: [node("u1", "user", null), node("a1", "assistant", "u1")],
      selected_content: [
        { id: "u1", revision: "1", message: "Question", images: [] },
        { id: "a1", revision: "1", message: "Answer", images: [] }
      ],
      view,
      purpose: "send",
      storage_context_digest: "storage"
    }
  }),
  fence: () => () => true
})

const visible = [
  { id: "u1", isBot: false, role: "user", name: "You", message: "Question", sources: [] },
  {
    id: "a1",
    isBot: true,
    role: "assistant",
    name: "Assistant",
    message: "Answer",
    sources: [],
    metadataExtra: { tldw_generation_v1: { status: "length" } }
  }
]

const renderBody = () => {
  rowProps.clear()
  sidepanelMessageState.value = { ...sidepanelMessageState.value, messages: visible }
  return render(<SidePanelBody />)
}

describe("SidePanelBody Continue and saved-status actions (CM-03)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetSavedReplyRefusals()
    selection.current = null
  })

  it("offers Continue and the recorded-status actions on the last saved reply of a server chat", async () => {
    selection.current = savedChat()
    renderBody()
    const last = rowProps.get("a1")!
    expect(last.savedStatusActions).toBe(true)
    expect(last.onContinue).toEqual(expect.any(Function))
    // Only the reply being continued carries the recorded-status actions.
    expect(rowProps.get("u1")!.savedStatusActions).toBe(false)

    await act(async () => {
      ;(last.onContinue as () => void)()
    })
    expect(sidepanelMessageState.value.continueLastMessage).toHaveBeenCalledTimes(1)
  })

  it("offers Continue in a chat without a selected saved history, without the recorded-status actions", () => {
    sidepanelMessageState.value = { ...sidepanelMessageState.value, serverChatId: null }
    renderBody()
    const last = rowProps.get("a1")!
    expect(last.onContinue).toEqual(expect.any(Function))
    expect(last.savedStatusActions).toBe(false)
    sidepanelMessageState.value = { ...sidepanelMessageState.value, serverChatId: "chat-1" }
  })

  it("withholds Continue where the full page does: a device-owned saved history", () => {
    selection.current = savedChat({ kind: "local" })
    renderBody()
    expect(rowProps.get("a1")!.onContinue).toBeUndefined()
    expect(rowProps.get("a1")!.savedStatusActions).toBe(false)
  })

  it("withdraws Continue after the server refuses to continue that reply", () => {
    selection.current = savedChat()
    const refused: NativeHistoryRegenerate = {
      owner_key: "owner",
      conversation_id: "chat-1",
      message_id: "a1",
      message_revision: "1",
      input_message_id: "u1",
      input_message_revision: "1",
      action: "continue",
      isCurrent: () => true,
      isOriginCurrent: () => true
    }
    rememberSavedReplyRefusal(refused, { status: 409, details: { detail: { reason: "reply" } } })
    renderBody()
    const last = rowProps.get("a1")!
    expect(last.onContinue).toBeUndefined()
    // Retry still runs: the refusal covered only Continue.
    expect(last.savedStatusActions).toBe(true)
  })
})

describe("SidePanelBody saved-reply refusals (review follow-up)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetSavedReplyRefusals()
  })

  it("disables Retry and withdraws Continue after the server refuses the saved question", () => {
    selection.current = savedChat()
    const refused: NativeHistoryRegenerate = {
      owner_key: "owner",
      conversation_id: "chat-1",
      message_id: "a1",
      message_revision: "1",
      input_message_id: "u1",
      input_message_revision: "1",
      action: "regenerate",
      isCurrent: () => true,
      isOriginCurrent: () => true
    }
    rememberSavedReplyRefusal(refused, { status: 409, details: { detail: { reason: "question" } } })
    renderBody()
    const last = rowProps.get("a1")!
    expect(last.onContinue).toBeUndefined()
    expect(last.regenerateDisabledReason).toMatch(/can't be regenerated here/)
  })
})

describe("SidePanelBody alternative replies (CM-N2)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetSavedReplyRefusals()
  })

  it("opens the newest message on the chosen alternative's branch, not just the alternative", async () => {
    const choose = vi.fn(async () => true)
    const nodes = [
      node("u1", "user", null),
      node("a1", "assistant", "u1"),
      node("u2", "user", "a1"),
      node("a2", "assistant", "u2"),
      node("a1b", "assistant", "u1")
    ]
    const branchView = { ...view, cursor: { kind: "after_message" as const, message_id: "a1b" } }
    const state = {
      status: "ready",
      owner: { kind: "native" },
      view: branchView,
      capture: { status: "captured", snapshot: { nodes }, rows: [], selected_content: [], view: branchView }
    }
    selection.current = { ...state, getCurrent: () => state, fence: () => () => true, choose }
    rowProps.clear()
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      messages: [
        { id: "u1", isBot: false, role: "user", name: "You", message: "Question", sources: [] },
        {
          id: "a1b", isBot: true, role: "assistant", name: "Assistant", message: "Newer answer", sources: [],
          variants: [{ id: "a1", message: "Older answer" }, { id: "a1b", message: "Newer answer" }],
          activeVariantIndex: 1
        }
      ]
    }
    render(<SidePanelBody />)
    await act(async () => {
      ;(rowProps.get("a1b")!.onSwipePrev as () => void)()
    })
    expect(choose).toHaveBeenCalledWith({ kind: "after_message", message_id: "a2" })
  })
})
