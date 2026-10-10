// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundMessage } from "../Message"
import { CompactMessage } from "../CompactMessage"
import { ReasoningBlock } from "../ReasoningBlock"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import { resolveCharacterMessageType } from "@/utils/character-greetings"

const reasoningPresentation = vi.hoisted(() => ({ actual: false }))
const markdownCalls = vi.hoisted(() => [] as Array<Record<string, unknown>>)
const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))
const parseReasoningMock = vi.hoisted(() =>
  vi.fn((content: string) => [{ type: "message", content }])
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key,
    i18n: { language: "en" }
  })
}))

vi.mock("antd", async importOriginal => {
  const actual = await importOriginal<typeof import("antd")>()
  const ModalMock = ({
    open,
    children
  }: {
    open?: boolean
    children?: React.ReactNode
  }) => (open ? <div>{children}</div> : null)
  const messageApi = {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }

  return {
    App: {
      useApp: () => ({ message: messageApi })
    },
    Tag: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
    Image: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Collapse: (props: React.ComponentProps<typeof actual.Collapse>) => reasoningPresentation.actual
      ? <actual.Collapse {...props} />
      : <div>
          {props.items?.map((item, index) => (
            <section key={index}>
              <div>{item.label}</div>
              {item.children}
            </section>
          )) ?? props.children}
        </div>,
    Avatar: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Modal: Object.assign(ModalMock, {
      confirm: vi.fn()
    }),
    message: messageApi
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.values.has(key) ? storageState.values.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("@/components/Common/Markdown", () => ({
  default: (props: Record<string, unknown>) => {
    markdownCalls.push(props)
    return <div data-testid="mock-markdown">{String(props.message ?? "")}</div>
  }
}))

vi.mock("../ActionInfo", () => ({
  LoadingStatus: () => null
}))

vi.mock("../EditMessageForm", () => ({
  EditMessageForm: ({ value }: { value: string }) => <textarea aria-label="Edit canonical message" defaultValue={value} />
}))

vi.mock("../PlaygroundUserMessage", () => ({
  PlaygroundUserMessageBubble: ({ message }: { message: string }) => (
    <div>{message}</div>
  )
}))

vi.mock("@/components/Sidepanel/Chat/FeedbackModal", () => ({
  FeedbackModal: () => null
}))

vi.mock("@/components/Sidepanel/Chat/SourceFeedback", () => ({
  SourceFeedback: () => null
}))

vi.mock("@/components/Sidepanel/Chat/ToolCallBlock", () => ({
  ToolCallBlock: () => null
}))

vi.mock("../MessageActionsBar", () => ({
  MessageActionsBar: ({ onEdit }: { onEdit?: () => void }) => <button type="button" onClick={onEdit}>Edit canonical message</button>
}))

vi.mock("../DiscoSkillAnnotation", () => ({
  DiscoSkillAnnotation: () => null
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    cancel: vi.fn(),
    isSpeaking: false,
    speak: vi.fn()
  })
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [false, vi.fn()]
}))

vi.mock("@/hooks/useFeedback", () => ({
  useFeedback: () => ({
    thumb: null,
    detail: "",
    sourceFeedback: {},
    canSubmit: false,
    isSubmitting: false,
    showThanks: false,
    submitThumb: vi.fn(),
    submitDetail: vi.fn(),
    submitSourceThumb: vi.fn()
  })
}))

vi.mock("@/hooks/useImplicitFeedback", () => ({
  useImplicitFeedback: () => ({
    trackCopy: vi.fn(),
    trackSourcesExpanded: vi.fn(),
    trackSourceClick: vi.fn(),
    trackCitationUsed: vi.fn(),
    trackDwellTime: vi.fn()
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasFeedbackExplicit: false,
      hasFeedbackImplicit: false
    }
  })
}))

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({
    healthState: "ready",
    voicesAvailable: true
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    saveChatKnowledge: vi.fn(async () => undefined),
    createChatCompletion: vi.fn(async () => ({}))
  }
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: any) => unknown) =>
    selector({
      setReplyTarget: vi.fn(),
      ragPinnedResults: [],
      setMessages: vi.fn()
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector: (state: any) => unknown) =>
    selector({
      apiProvider: "openai",
      updateSetting: vi.fn()
    })
}))

vi.mock("@/store", () => ({
  useStoreMessage: (selector: (state: any) => unknown) =>
    selector({
      selectedModel: "gpt-4o-mini"
    })
}))

vi.mock("@/hooks/useDiscoSkills", () => ({
  useDiscoSkills: () => ({
    enabled: false,
    stats: null,
    triggerProbabilityBase: 0,
    persistComments: false
  })
}))

vi.mock("@/libs/reasoning", async importOriginal => ({
  ...await importOriginal<typeof import("@/libs/reasoning")>(),
  parseReasoning: parseReasoningMock
}))

vi.mock("@/utils/chat-error-message", () => ({
  decodeChatErrorPayload: vi.fn(() => null)
}))

vi.mock("@/utils/feedback", () => ({
  getSourceFeedbackKey: () => "feedback-key"
}))

vi.mock("@/utils/clipboard", () => ({
  copyToClipboard: vi.fn(async () => undefined)
}))

vi.mock("@/utils/chat-style", () => ({
  buildChatTextClass: () => ""
}))

vi.mock("@/utils/text-highlight", () => ({
  highlightText: (value: string) => value
}))

vi.mock("@/db/dexie/models", () => ({
  removeModelSuffix: (value: string) => value
}))

vi.mock("@/utils/color", () => ({
  tagColors: {}
}))

vi.mock("@/utils/disco-skill-check", () => ({
  attemptSkillTrigger: vi.fn(),
  buildSkillPrompt: vi.fn(() => ""),
  createSkillComment: vi.fn(() => null)
}))

vi.mock("@/utils/character-mood", () => ({
  detectCharacterMood: () => ({ label: null }),
  normalizeCharacterMoodLabel: (value: unknown) => value,
  resolveCharacterBaseAvatarUrl: () => "",
  resolveCharacterMoodImageUrl: () => ""
}))

vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))
vi.mock("@/db/dexie/helpers", async importOriginal => ({
  ...await importOriginal<typeof import("@/db/dexie/helpers")>(),
  updateMessageDiscoSkillComment: vi.fn(async () => undefined)
}))

vi.mock("../message-layout", () => ({
  resolveAvatarColumnAlignment: () => "",
  resolveMessageRenderSide: () => "left"
}))

vi.mock("../playground-message-shortcuts", () => ({
  resolvePlaygroundMessageShortcutAction: () => null
}))

vi.mock("../quick-message-actions", () => ({
  buildQuickMessageActionPrompt: vi.fn(() => "prompt")
}))

const baseProps: React.ComponentProps<typeof PlaygroundMessage> = {
  message: "```mermaid\ngraph TD\n  A-->B\n```",
  isBot: true,
  role: "assistant",
  name: "Assistant",
  currentMessageIndex: 0,
  totalMessages: 1,
  onRegenerate: vi.fn(),
  onContinue: vi.fn(),
  onEditFormSubmit: vi.fn(),
  isProcessing: false,
  isStreaming: false,
  conversationInstanceId: "conversation-1"
}

describe("PlaygroundMessage Mermaid rendering gates", () => {
  beforeEach(() => {
    markdownCalls.length = 0
    storageState.values.clear()
    parseReasoningMock.mockImplementation((content: string) => [
      { type: "message", content }
    ])
  })

  it("projects saved raw greetings with distinct names and keeps canonical content unchanged on reload", async () => {
    const raw = "{{user}} meets {{char}}\n  exact  "
    storageState.values.set("chatUserDisplayName", "Viewer $&")
    const capture = {
      status: "captured", view: { conversation_id: "saved-chat" },
      snapshot: { nodes: [{ id: "saved-root", role: "assistant", parent_id: null }] },
      rows: [{ id: "saved-root", role: "assistant", parent_id: null }],
      selected_content: [{ id: "saved-root", message: raw, images: [] }]
    } as unknown as import("@/types/history-selection").HistorySelectionCaptureV1
    const display = formatSelectedHistory(capture)
    const root = display.messages[0]
    const props = { ...baseProps, message: root.message,
      message_type: resolveCharacterMessageType(root, 0, true), serverChatId: "saved-chat",
      characterIdentityEnabled: true, characterIdentity: { id: "12", name: "Guide $'" } }
    const view = render(<PlaygroundMessage {...props} />)
    await waitFor(() => expect(markdownCalls.at(-1)?.message).toBe("Viewer $& meets Guide $'\n  exact  "))
    storageState.values.set("chatUserDisplayName", "")
    view.rerender(<PlaygroundMessage {...props} characterIdentity={{ id: "12", name: "Updated Guide" }} />)
    await waitFor(() => expect(markdownCalls.at(-1)?.message).toBe("User meets Updated Guide\n  exact  "))
    expect(props.message).toBe(raw)
    expect(capture.selected_content[0].message).toBe(raw)
    expect(display.history[0].content).toBe(raw)
    fireEvent.click(screen.getByRole("button", { name: "Edit canonical message" }))
    expect(screen.getByRole("textbox", { name: "Edit canonical message" })).toHaveValue(raw)
    expect(root.serverMessageId).toBeUndefined()
    expect(root.serverMessageVersion).toBeUndefined()
  })

  it("keeps a mismatched character name unresolved in a saved greeting", async () => {
    render(<PlaygroundMessage {...baseProps} message="{{user}} meets {{char}}" message_type="character:greeting"
      characterIdentityEnabled characterIdentity={{ id: "12", name: "Other Guide" }} speakerCharacterId="34" />)
    await waitFor(() => expect(markdownCalls.at(-1)?.message).toBe("User meets {{char}}"))
  })
  it("does not substitute placeholders in ordinary assistant content", async () => {
    const raw = "{{user}} meets {{char}}"
    render(<PlaygroundMessage {...baseProps} message={raw} characterIdentityEnabled characterIdentity={{ id: "12", name: "Guide" }} />)
    await waitFor(() => expect(markdownCalls.at(-1)?.message).toBe(raw))
  })

  it("enables Mermaid for a completed assistant markdown message by default", async () => {
    render(<PlaygroundMessage {...baseProps} />)

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        enableMermaidArtifactActions: true,
        enableMermaidDiagrams: true
      })
    )
  })

  it("uses the saved assistant message id as the Mermaid artifact context", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        messageId="assistant-message-123"
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        artifactContextId: "assistant-message-123-segment-0",
        enableMermaidArtifactActions: true,
        enableMermaidDiagrams: true
      })
    )
  })

  it("uses a safe local Mermaid artifact context when no message or conversation id is available", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        conversationInstanceId={undefined as any}
      />
    )

    await screen.findByTestId("mock-markdown")

    const contextId = String(markdownCalls[0]?.artifactContextId ?? "")
    expect(contextId).toBe("local-message-0-segment-0")
    expect(contextId).not.toContain("undefined")
    expect(contextId).not.toContain("null")
  })

  it("namespaces Mermaid artifact contexts per assistant Markdown segment", async () => {
    parseReasoningMock.mockImplementation(() => [
      { type: "message", content: "```mermaid\ngraph TD\n  A-->B\n```" },
      { type: "message", content: "```mermaid\ngraph TD\n  A-->B\n```" }
    ])

    render(<PlaygroundMessage {...baseProps} messageId="assistant-message-123" />)

    await waitFor(() => {
      expect(markdownCalls).toHaveLength(2)
    })

    expect(markdownCalls.map((call) => call.artifactContextId)).toEqual([
      "assistant-message-123-segment-0",
      "assistant-message-123-segment-1"
    ])
  })

  it("uses plain text instead of Markdown while assistant output is streaming", () => {
    render(<PlaygroundMessage {...baseProps} isStreaming />)

    expect(
      screen.getByTestId("playground-streaming-plain-text")
    ).toHaveTextContent("graph TD")
    expect(markdownCalls).toHaveLength(0)
  })

  it("keeps Mermaid enabled for an older completed assistant message while another row streams", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        currentMessageIndex={0}
        totalMessages={2}
        isStreaming
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        enableMermaidArtifactActions: true,
        enableMermaidDiagrams: true
      })
    )
  })

  it("does not enable Mermaid on user messages", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        isBot={false}
        role="user"
        name="User"
      />
    )

    await waitFor(() => {
      expect(markdownCalls).toHaveLength(0)
    })
  })

  it("does not enable Mermaid when the chat setting is disabled", async () => {
    storageState.values.set("renderMermaidDiagrams", false)

    render(<PlaygroundMessage {...baseProps} />)

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]?.enableMermaidDiagrams).not.toBe(true)
  })

  it("forwards Mermaid enablement through completed assistant reasoning", async () => {
    render(
      <ReasoningBlock
        content="```mermaid\ngraph TD\n  A-->B\n```"
        isStreaming={false}
        reasoningRunning={false}
        assistantTextClass=""
        markdownBaseClasses=""
        t={((_key: string, fallback?: string) => fallback || _key) as any}
        enableMermaidDiagrams
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        enableMermaidDiagrams: true
      })
    )
    expect(markdownCalls[0]?.enableMermaidArtifactActions).not.toBe(true)
  })

  it("keeps Mermaid disabled while reasoning is actively streaming", async () => {
    render(
      <ReasoningBlock
        content="```mermaid\ngraph TD\n  A-->B\n```"
        isStreaming
        reasoningRunning
        assistantTextClass=""
        markdownBaseClasses=""
        t={((_key: string, fallback?: string) => fallback || _key) as any}
        enableMermaidDiagrams
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]?.enableMermaidDiagrams).not.toBe(true)
  })

  it("keeps Mermaid enabled for completed compact rows while a later row streams", async () => {
    render(
      <CompactMessage
        message="```mermaid\ngraph TD\n  A-->B\n```"
        isBot
        name="Assistant"
        currentMessageIndex={0}
        totalMessages={2}
        isStreaming
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        enableMermaidArtifactActions: true,
        enableMermaidDiagrams: true
      })
    )
  })

  it("does not compute a compact Mermaid artifact context when Mermaid actions are disabled", async () => {
    render(
      <CompactMessage
        message="```mermaid\ngraph TD\n  A-->B\n```"
        isBot
        name="Assistant"
        currentMessageIndex={0}
        totalMessages={1}
        isStreaming
      />
    )

    await screen.findByTestId("mock-markdown")

    expect(markdownCalls[0]).toEqual(
      expect.objectContaining({
        enableMermaidArtifactActions: false,
        enableMermaidDiagrams: false
      })
    )
    expect(markdownCalls[0]?.artifactContextId).toBeUndefined()
  })
})


describe("TASK-13274 actual streaming reasoning disclosure", () => {
  beforeEach(async () => {
    reasoningPresentation.actual = true
    markdownCalls.length = 0
    storageState.values.clear()
    const actual = await vi.importActual<typeof import("@/libs/reasoning")>("@/libs/reasoning")
    parseReasoningMock.mockImplementation(actual.parseReasoning)
  })
  afterEach(() => { reasoningPresentation.actual = false })

  it("shows Thinking before the first answer with actual Collapse closed by default", async () => {
    render(<PlaygroundMessage {...baseProps} message="<think>Original thought" isStreaming />)
    const disclosure = await screen.findByRole("button", { name: /Thinking…$/ })
    expect(disclosure).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByText("Original thought")).toBeNull()
    expect(screen.queryByTestId("playground-streaming-plain-text")).toBeNull()
    expect(markdownCalls).toHaveLength(0)
  })

  it("keeps the user's disclosure choice while reasoning grows and answer streaming starts", async () => {
    const mounted = render(<PlaygroundMessage {...baseProps} message="<think>Original thought" isStreaming />)
    const disclosure = await screen.findByRole("button", { name: /Thinking…$/ })
    fireEvent.click(disclosure)
    await screen.findByText("Original thought")
    expect(disclosure).toHaveAttribute("aria-expanded", "true")
    expect(markdownCalls.every(call => call.enableMermaidDiagrams !== true)).toBe(true)
    mounted.rerender(<PlaygroundMessage {...baseProps} message="<think>Original thought then more" isStreaming />)
    expect(disclosure).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(disclosure)
    mounted.rerender(<PlaygroundMessage {...baseProps} message="<think>Original thought then more</think>Final answer" isStreaming />)
    const completedDisclosure = screen.getByRole("button", { name: /Model's reasoning \(optional\)$/ })
    expect(completedDisclosure).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByTestId("playground-streaming-plain-text")).toHaveTextContent("Final answer")
    expect(screen.getByTestId("playground-streaming-plain-text").textContent).toBe("Final answer")
    expect(markdownCalls.every(call => call.message !== "Final answer")).toBe(true)
  })

  it("respects opt-in open reasoning and still lets the user collapse active progress", async () => {
    const props = { content: "Original thought", isStreaming: true, reasoningRunning: true,
      openReasoning: true, assistantTextClass: "", markdownBaseClasses: "",
      t: ((_key: string, fallback?: string) => fallback || _key) as React.ComponentProps<typeof ReasoningBlock>["t"] }
    const mounted = render(<ReasoningBlock {...props} />)
    const disclosure = screen.getByRole("button", { name: /Thinking…$/ })
    expect(disclosure).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(disclosure)
    expect(disclosure).toHaveAttribute("aria-expanded", "false")
    mounted.rerender(<ReasoningBlock {...props} content="Original thought then more" />)
    expect(disclosure).toHaveAttribute("aria-expanded", "false")
  })

  it("keeps a plain answer's exact lightweight renderer without an empty disclosure", () => {
    render(<PlaygroundMessage {...baseProps} message={"  Plain answer\n"} isStreaming />)
    expect(screen.getByTestId("playground-streaming-plain-text").textContent).toBe("  Plain answer\n")
    expect(screen.queryByRole("button", { name: /Thinking…$/ })).toBeNull()
    expect(screen.queryByRole("button", { name: /Model's reasoning \(optional\)$/ })).toBeNull()
    expect(markdownCalls).toHaveLength(0)
  })
})
