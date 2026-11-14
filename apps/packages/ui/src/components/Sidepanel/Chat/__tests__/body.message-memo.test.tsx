// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidePanelBody } from "../body"

// Per-row render counts for the memoized row spy, keyed by messageId.
const rowRenderCounts = vi.hoisted(() => new Map<string, number>())
const sidepanelMessageState = vi.hoisted(() => ({
  value: {
    messages: [] as any[],
    setMessages: vi.fn(),
    setHistory: vi.fn(),
    streaming: false,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    isSearchingInternet: false,
    createChatBranch: vi.fn(),
    historyId: "history-1",
    temporaryChat: false,
    stopStreamingRequest: vi.fn(),
    serverChatId: null,
    isEmbedding: false
  }
}))

// Production's useMessage re-creates its action callbacks every render
// (deleteMessage's useCallback closes over the current `messages` snapshot;
// editMessage is a per-render factory product). The mock therefore hands
// body.tsx NEW deleteMessage/editMessage identities on every useMessage()
// call, so the row-memo contract below must hold under that per-render
// churn — not just for identity-stable mocks.
vi.mock("~/hooks/useMessage", () => ({
  useMessage: () => ({
    ...sidepanelMessageState.value,
    deleteMessage: vi.fn(),
    editMessage: vi.fn()
  })
}))

// Memoized row spy standing in for the real PlaygroundMessage: default
// shallow prop equality, the same contract the real React.memo boundary
// (with its comparator) enforces for the props body.tsx passes.
vi.mock("~/components/Common/Playground/Message", () => ({
  PlaygroundMessage: React.memo(function PlaygroundMessageSpy(props: {
    messageId?: string
    isStreaming: boolean
    message: string
  }) {
    const key = props.messageId ?? "unknown"
    rowRenderCounts.set(key, (rowRenderCounts.get(key) ?? 0) + 1)
    return (
      <article
        data-testid="sidepanel-row"
        data-is-streaming={props.isStreaming ? "true" : "false"}
      >
        {props.message}
      </article>
    )
  })
}))

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 120,
    getVirtualItems: () =>
      Array.from({ length: count }, (_unused, index) => ({
        index,
        key: `row-${index}`,
        start: index * 120
      })),
    measureElement: vi.fn(),
    scrollToIndex: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/store/webui", () => ({
  useWebUI: () => ({ ttsEnabled: false })
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))

vi.mock("@/store/option", () => {
  // Dev's body.tsx routes Stop-button focus through useExplicitStopFocus,
  // which reads the draft snapshot via useStoreMessageOption.getState() (and
  // subscribes on stop); the memo harness keeps those inert.
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
      (selector: (state: Record<string, unknown>) => unknown) =>
        selector(optionState),
      {
        getState: () => optionState,
        subscribe: () => () => {}
      }
    )
  }
})

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => null
}))

vi.mock("../empty", () => ({
  EmptySidePanel: () => <div data-testid="empty-sidepanel" />
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "generated-id"
}))

const buildMessages = (count: number) =>
  Array.from({ length: count }, (_unused, index) => ({
    id: `msg-${index}`,
    isBot: index % 2 === 1,
    role: index % 2 === 1 ? "assistant" : "user",
    name: index % 2 === 1 ? "Assistant" : "User",
    message:
      index === count - 1
        ? "final assistant reply"
        : `message content ${index}`,
    messageType: undefined,
    serverMessageId: null
  }))

// Stage 1 semantics: a streaming flush replaces only the last message's
// entry; every other message object keeps referential identity.
const replaceLastMessage = (
  messages: any[],
  message: string
) => {
  const next = messages.slice()
  next[next.length - 1] = { ...next[next.length - 1], message }
  return next
}

const countRowsRendered = () =>
  Array.from(rowRenderCounts.values()).filter((count) => count > 0).length

describe("SidePanelBody row memoization (WP-02/WP-03/WP-09)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    rowRenderCounts.clear()
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      messages: [],
      streaming: false,
      isProcessing: false
    }
  })

  it("a stream start plus token flush re-renders only the last row, not the other 49", () => {
    const initialMessages = buildMessages(50)
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      messages: initialMessages
    }

    const { rerender } = render(<SidePanelBody />)

    expect(screen.getAllByTestId("sidepanel-row")).toHaveLength(50)
    expect(countRowsRendered()).toBe(50)

    rowRenderCounts.clear()

    // Stream start: `streaming` flips true and the last message entry is
    // replaced with the first partial chunk.
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      streaming: true,
      messages: replaceLastMessage(initialMessages, "partial ")
    }
    rerender(<SidePanelBody />)

    // Guard: the last row DID re-render...
    expect(rowRenderCounts.get("msg-49")).toBe(1)
    // ...and no sibling row re-rendered.
    for (let index = 0; index < 49; index += 1) {
      expect(rowRenderCounts.get(`msg-${index}`) ?? 0).toBe(0)
    }

    rowRenderCounts.clear()

    // Token flush while streaming: only the last row's text changes.
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      messages: replaceLastMessage(initialMessages, "partial streamed text")
    }
    rerender(<SidePanelBody />)

    expect(rowRenderCounts.get("msg-49")).toBe(1)
    for (let index = 0; index < 49; index += 1) {
      expect(rowRenderCounts.get(`msg-${index}`) ?? 0).toBe(0)
    }
    expect(screen.getByText("partial streamed text")).toBeInTheDocument()
  })

  it("keeps sibling rows memo-stable when the hook re-creates delete/edit identities per render", () => {
    const initialMessages = buildMessages(50)
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      streaming: true,
      messages: initialMessages
    }

    const { rerender } = render(<SidePanelBody />)

    expect(screen.getAllByTestId("sidepanel-row")).toHaveLength(50)
    rowRenderCounts.clear()

    // Token flush: the useMessage mock hands body.tsx NEW deleteMessage and
    // editMessage identities on this render (production re-creates both per
    // streaming flush). Sibling rows must still not re-render — row props
    // must not depend on hook callback identity.
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      messages: replaceLastMessage(initialMessages, "next token")
    }
    rerender(<SidePanelBody />)

    expect(rowRenderCounts.get("msg-49")).toBe(1)
    for (let index = 0; index < 49; index += 1) {
      expect(rowRenderCounts.get(`msg-${index}`) ?? 0).toBe(0)
    }
    expect(screen.getByText("next token")).toBeInTheDocument()
  })

  it("narrows isStreaming so only the last row receives isStreaming=true while streaming", () => {
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      streaming: true,
      messages: buildMessages(6)
    }

    render(<SidePanelBody />)

    const rows = screen.getAllByTestId("sidepanel-row")
    expect(rows).toHaveLength(6)
    expect(
      rows.filter((row) => row.getAttribute("data-is-streaming") === "true")
    ).toHaveLength(1)
    // The streaming row is the last one.
    expect(rows[5]).toHaveAttribute("data-is-streaming", "true")
    for (let index = 0; index < 5; index += 1) {
      expect(rows[index]).toHaveAttribute("data-is-streaming", "false")
    }
  })

  it("clears the streaming flag on every row once streaming ends", () => {
    const messages = buildMessages(4)
    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      streaming: true,
      messages
    }

    const { rerender } = render(<SidePanelBody />)

    expect(
      screen
        .getAllByTestId("sidepanel-row")
        .filter((row) => row.getAttribute("data-is-streaming") === "true")
    ).toHaveLength(1)

    sidepanelMessageState.value = {
      ...sidepanelMessageState.value,
      streaming: false
    }
    rerender(<SidePanelBody />)

    expect(
      screen
        .getAllByTestId("sidepanel-row")
        .filter((row) => row.getAttribute("data-is-streaming") === "true")
    ).toHaveLength(0)
  })
})
