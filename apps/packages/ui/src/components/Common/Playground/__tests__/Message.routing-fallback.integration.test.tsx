// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { PlaygroundMessage } from "../Message"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const storageOverrides = vi.hoisted(() => new Map<string, unknown>())
const detectCharacterMoodMock = vi.hoisted(() =>
  vi.fn(() => ({ label: "neutral", confidence: 0.5, topic: null }))
)
const resolveCharacterMoodImageUrlMock = vi.hoisted(() =>
  vi.fn((_character: unknown, _moodLabel: unknown) => "")
)
const initializeMock = vi.hoisted(() => vi.fn(async () => undefined))
const saveChatKnowledgeMock = vi.hoisted(() => vi.fn(async () => undefined))
const staticMessageMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn()
}))
const contextMessageMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string, options?: Record<string, unknown>) => {
      const template = fallback || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("antd", () => ({
  Tag: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  Image: ({ src, alt }: { src?: string; alt?: string }) => (
    <img src={src || ""} alt={alt || ""} />
  ),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Collapse: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Avatar: ({ src, alt }: { src?: string; alt?: string }) => (
    <img src={src || ""} alt={alt || ""} />
  ),
  Modal: ({
    open,
    children
  }: {
    open?: boolean
    children: React.ReactNode
  }) => (open ? <div>{children}</div> : null),
  App: {
    useApp: () => ({ message: contextMessageMock })
  },
  message: staticMessageMock
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageOverrides.has(key) ? storageOverrides.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [
    storageOverrides.has("chatShowMoodBadge")
      ? Boolean(storageOverrides.get("chatShowMoodBadge"))
      : false,
    vi.fn(),
    { isLoading: false, setRenderValue: vi.fn() }
  ]
}))

vi.mock("@/components/Common/Markdown", () => ({
  default: ({ message }: { message: string }) => (
    <div data-testid="mock-markdown">{message}</div>
  )
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
  MessageActionsBar: ({
    onSaveKnowledge
  }: {
    onSaveKnowledge?: (makeFlashcard: boolean) => void
  }) => (
    <div data-testid="message-actions">
      <button type="button" onClick={() => onSaveKnowledge?.(false)}>
        Save to Notes
      </button>
    </div>
  )
}))

vi.mock("../ReasoningBlock", () => ({
  ReasoningBlock: ({ content }: { content: string }) => <div>{content}</div>
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
    initialize: initializeMock,
    saveChatKnowledge: saveChatKnowledgeMock,
    createChatCompletion: vi.fn(async () => ({}))
  }
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: {
      setReplyTarget: (...args: unknown[]) => void
      ragPinnedResults: unknown[]
      setMessages: (...args: unknown[]) => void
    }) => unknown
  ) =>
    selector({
      setReplyTarget: vi.fn(),
      ragPinnedResults: [],
      setMessages: vi.fn()
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (
    selector: (state: { apiProvider: string | null; updateSetting: (...args: unknown[]) => void }) => unknown
  ) =>
    selector({
      apiProvider: null,
      updateSetting: vi.fn()
    })
}))

vi.mock("@/store", () => ({
  useStoreMessage: (selector: (state: { selectedModel: string }) => unknown) =>
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
  parseReasoning: (content: string) => [{ type: "message", content }]
}))

vi.mock("@/utils/chat-error-message", () => ({
  decodeChatErrorPayload: () => null
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
  detectCharacterMood: detectCharacterMoodMock,
  normalizeCharacterMoodLabel: (value: unknown) =>
    value === "neutral" ? "neutral" : null,
  resolveCharacterBaseAvatarUrl: () => "",
  resolveCharacterMoodImageUrl: resolveCharacterMoodImageUrlMock
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
  message: "Sample assistant output",
  isBot: true,
  role: "assistant",
  name: "Assistant",
  currentMessageIndex: 0,
  totalMessages: 1,
  onRegenerate: vi.fn(),
  onEditFormSubmit: vi.fn(),
  isProcessing: false,
  isStreaming: false,
  conversationInstanceId: "conversation-1"
}

describe("PlaygroundMessage routing fallback integration", () => {
  beforeEach(() => {
    storageOverrides.clear()
    initializeMock.mockClear()
    saveChatKnowledgeMock.mockClear()
    staticMessageMock.success.mockClear()
    staticMessageMock.error.mockClear()
    contextMessageMock.success.mockClear()
    contextMessageMock.error.mockClear()
    resolveCharacterMoodImageUrlMock.mockReset()
    resolveCharacterMoodImageUrlMock.mockReturnValue("")
    detectCharacterMoodMock.mockReset()
    detectCharacterMoodMock.mockReturnValue({
      label: "neutral",
      confidence: 0.5,
      topic: null
    })
  })

  it("renders fallback routing audit metadata when generation info includes routing details", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{
          routing_policy: "auto-fallback",
          requested_provider: "openai",
          requested_model: "gpt-4o-mini",
          resolved_provider: "anthropic",
          resolved_model: "claude-3-5-sonnet",
          routing_attempts: 2,
          fallback_reason: "Rate limited upstream"
        }}
      />
    )

    const audit = screen.getByTestId("message-fallback-audit")
    expect(audit).toHaveTextContent("Using: anthropic/claude-3-5-sonnet")
    expect(audit).toHaveTextContent("(fallback)")
    expect(audit).toHaveTextContent("2 attempts")
    expect(audit).toHaveTextContent("Rate limited upstream")
  })

  it("does not render fallback audit metadata when routing details are absent", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{
          usage: {
            prompt_tokens: 20,
            completion_tokens: 10
          }
        }}
      />
    )

    expect(screen.queryByTestId("message-fallback-audit")).toBeNull()
  })

  it("hides mood badge when chat mood visibility is disabled", () => {
    storageOverrides.set("chatShowMoodBadge", false)

    render(<PlaygroundMessage {...baseProps} />)

    expect(screen.queryByTestId("message-mood-indicator")).toBeNull()
  })

  it("hides mood badge by default", () => {
    render(<PlaygroundMessage {...baseProps} />)

    expect(screen.queryByTestId("message-mood-indicator")).toBeNull()
  })

  it("shows mood badge when chat mood visibility is enabled", () => {
    storageOverrides.set("chatShowMoodBadge", true)

    render(<PlaygroundMessage {...baseProps} />)

    const moodBadge = screen.getByTestId("message-mood-indicator")
    expect(moodBadge).toHaveTextContent("Mood: neutral")
  })

  it("formats hyphenated mood labels in the badge", () => {
    storageOverrides.set("chatShowMoodBadge", true)

    render(
      <PlaygroundMessage
        {...baseProps}
        moodLabel="Thinking Hard"
      />
    )

    expect(screen.getByTestId("message-mood-indicator")).toHaveTextContent(
      "Mood: thinking hard"
    )
  })

  it("defaults mood confidence to off for non-character chats", () => {
    storageOverrides.set("chatShowMoodBadge", true)

    render(
      <PlaygroundMessage
        {...baseProps}
        moodLabel="neutral"
        moodConfidence={0.73}
      />
    )

    const moodBadge = screen.getByTestId("message-mood-indicator")
    expect(moodBadge).toHaveTextContent("Mood: neutral")
    expect(moodBadge).not.toHaveTextContent("(73%)")
  })

  it("shows mood confidence when enabled", () => {
    storageOverrides.set("chatShowMoodBadge", true)
    storageOverrides.set("chatShowMoodConfidence", true)

    render(
      <PlaygroundMessage
        {...baseProps}
        moodLabel="neutral"
        moodConfidence={0.73}
      />
    )

    const moodBadge = screen.getByTestId("message-mood-indicator")
    expect(moodBadge).toHaveTextContent("Mood: neutral (73%)")
  })

  it("hides mood confidence when disabled", () => {
    storageOverrides.set("chatShowMoodBadge", true)
    storageOverrides.set("chatShowMoodConfidence", false)

    render(
      <PlaygroundMessage
        {...baseProps}
        moodLabel="neutral"
        moodConfidence={0.73}
      />
    )

    const moodBadge = screen.getByTestId("message-mood-indicator")
    expect(moodBadge).toHaveTextContent("Mood: neutral")
    expect(moodBadge).not.toHaveTextContent("(73%)")
  })

  it("renders lightweight plain text while active assistant stream is running", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        isStreaming
        message={"streaming response▋"}
      />
    )

    expect(
      screen.getByTestId("playground-streaming-plain-text")
    ).toHaveTextContent("streaming response▋")
    expect(screen.queryByTestId("mock-markdown")).toBeNull()
  })

  it("renders markdown when stream is not active", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        isStreaming={false}
        isProcessing={false}
        message={"final response"}
      />
    )

    expect(await screen.findByTestId("mock-markdown")).toHaveTextContent(
      "final response"
    )
    expect(screen.queryByTestId("playground-streaming-plain-text")).toBeNull()
  })

  it("renders a custom explicit emote portrait for character messages", () => {
    const smugPortrait = "data:image/png;base64,c211Zw=="
    resolveCharacterMoodImageUrlMock.mockImplementation((_character, mood) =>
      mood === "smug" ? smugPortrait : ""
    )

    render(
      <PlaygroundMessage
        {...baseProps}
        moodLabel="smug"
        characterIdentityEnabled
        characterIdentity={
          {
            id: 42,
            name: "Ashley",
            extensions: {
              tldw: {
                mood_images: {
                  smug: smugPortrait
                }
              }
            }
          } as any
        }
        speakerCharacterId={42}
      />
    )

    expect(resolveCharacterMoodImageUrlMock).toHaveBeenCalledWith(
      expect.anything(),
      "smug"
    )
    expect(detectCharacterMoodMock).not.toHaveBeenCalled()
    expect(screen.getByAltText("Ashley")).toHaveAttribute("src", smugPortrait)
  })

  it("passes workspace scope when saving chat knowledge", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        serverChatId="workspace-chat-1"
        serverMessageId="message-1"
        scope={{ type: "workspace", workspaceId: "ws-1" }}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))

    await waitFor(() =>
      expect(tldwClient.saveChatKnowledge).toHaveBeenCalledWith(
        {
          conversation_id: "workspace-chat-1",
          message_id: "message-1",
          snippet: "Sample assistant output",
          make_flashcard: false
        },
        {
          scope: { type: "workspace", workspaceId: "ws-1" }
        }
      )
    )
    expect(contextMessageMock.success).toHaveBeenCalledWith("Saved to Notes")
    expect(staticMessageMock.success).not.toHaveBeenCalled()
  })
})
