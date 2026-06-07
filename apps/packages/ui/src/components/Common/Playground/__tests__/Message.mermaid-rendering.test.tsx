// @vitest-environment jsdom
import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundMessage } from "../Message"
import { CompactMessage } from "../CompactMessage"
import { ReasoningBlock } from "../ReasoningBlock"

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

vi.mock("antd", () => {
  const ModalMock = ({
    open,
    children
  }: {
    open?: boolean
    children?: React.ReactNode
  }) => (open ? <div>{children}</div> : null)

  return {
    Tag: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
    Image: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Collapse: ({
      children,
      items
    }: {
      children?: React.ReactNode
      items?: Array<{ children?: React.ReactNode; label?: React.ReactNode }>
    }) => (
      <div>
        {items?.map((item, index) => (
          <section key={index}>
            <div>{item.label}</div>
            {item.children}
          </section>
        )) ?? children}
      </div>
    ),
    Avatar: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Modal: Object.assign(ModalMock, {
      confirm: vi.fn()
    }),
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn()
    }
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
  EditMessageForm: () => <div data-testid="edit-form" />
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
  MessageActionsBar: () => null
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

vi.mock("@/libs/reasoning", () => ({
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

vi.mock("@/db/dexie/helpers", () => ({
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
        artifactContextId: "assistant-message-123",
        enableMermaidArtifactActions: true,
        enableMermaidDiagrams: true
      })
    )
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
})
