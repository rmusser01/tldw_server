import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidePanelBody } from "../body"

const deleteCallbacks = vi.hoisted(() => new Map<string, () => unknown>())

const sidepanelMessageState = vi.hoisted(() => ({
  value: {
    messages: [] as any[],
    setMessages: vi.fn(),
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

vi.mock("~/hooks/useMessage", () => ({
  useMessage: () => sidepanelMessageState.value
}))

vi.mock("~/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    dynamicUISurface?: string
    metadataExtra?: Record<string, any>
    message: string
    onDeleteMessage?: () => unknown
  }) => {
    if (props.onDeleteMessage) deleteCallbacks.set(props.message, props.onDeleteMessage)
    const envelope = props.metadataExtra?.dynamic_ui
    return (
      <article data-testid="sidepanel-message">
        {envelope ? (
          <pre
            data-testid="dynamic-ui-source-fallback"
            data-dynamic-ui-surface={props.dynamicUISurface}
          >
            {props.message}
          </pre>
        ) : (
          props.message
        )}
      </article>
    )
  }
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

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ serverChatCharacterId: null })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => null
}))

vi.mock("../empty", () => ({
  EmptySidePanel: () => <div data-testid="empty-sidepanel" />
}))

describe("SidePanelBody Dynamic UI fallback", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sidepanelMessageState.value.messages = []
  })

  it("renders OpenUI metadata as source fallback on the extension sidepanel surface", () => {
    sidepanelMessageState.value.messages = [
      {
        id: "assistant-openui",
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "root = <Card />",
        metadataExtra: {
          dynamic_ui: {
            renderer: "openui",
            version: "v1",
            source: "root = <Card />"
          }
        }
      }
    ]

    render(<SidePanelBody />)

    expect(screen.getByTestId("dynamic-ui-source-fallback")).toHaveTextContent(
      "root = <Card />"
    )
    expect(screen.getByTestId("dynamic-ui-source-fallback")).toHaveAttribute(
      "data-dynamic-ui-surface",
      "extension-sidepanel"
    )
    expect(screen.queryByTestId("openui-runtime")).not.toBeInTheDocument()
  })
})


it.each(["accepted", "refused", "rejected"])(
  "SidePanelBody returns the pending Delete %s outcome",
  async (outcome) => {
    deleteCallbacks.clear()
    sidepanelMessageState.value.messages = [
      {
        id: "saved",
        isBot: true,
        role: "assistant",
        name: "Assistant",
        message: "Saved reply"
      }
    ]
    let resolve!: (value: boolean) => void, reject!: (error: Error) => void
    sidepanelMessageState.value.deleteMessage.mockImplementation(
      () =>
        new Promise<boolean>((yes, no) => {
          resolve = yes
          reject = no
        })
    )
    render(<SidePanelBody />)
    const pending = deleteCallbacks.get("Saved reply")!()
    expect(pending).toBeInstanceOf(Promise)
    if (outcome === "rejected") {
      const assertion = expect(pending).rejects.toThrow(
        "message_has_descendants"
      )
      reject(new Error("message_has_descendants"))
      await assertion
    } else {
      resolve(outcome === "accepted")
      await expect(pending).resolves.toBe(outcome === "accepted")
    }
  }
)
