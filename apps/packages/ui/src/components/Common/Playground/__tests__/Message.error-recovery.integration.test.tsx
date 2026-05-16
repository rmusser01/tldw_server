// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { PlaygroundMessage } from "../Message"
import { IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE } from "@/utils/image-generation-chat"

const decodeChatErrorPayloadMock = vi.hoisted(() => vi.fn(() => null))
const updateChatModelSettingMock = vi.hoisted(() => vi.fn())
const antdMessageApi = vi.hoisted(() => ({
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
    Collapse: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Avatar: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Modal: Object.assign(ModalMock, {
      confirm: vi.fn()
    }),
    message: antdMessageApi
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
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
  MessageActionsBar: () => <div data-testid="message-actions" />
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
      apiProvider: "openai",
      updateSetting: updateChatModelSettingMock
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
  decodeChatErrorPayload: decodeChatErrorPayloadMock
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
  message: "Sample assistant output",
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

describe("PlaygroundMessage error recovery integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    decodeChatErrorPayloadMock.mockReturnValue(null)
  })

  it("wires retry/switch/fallback/continue actions for explicit provider errors", async () => {
    const user = userEvent.setup()
    const onRegenerate = vi.fn()
    const onContinue = vi.fn()
    const settingsEventSpy = vi.fn()
    window.addEventListener(
      "tldw:open-model-settings",
      settingsEventSpy as EventListener
    )

    decodeChatErrorPayloadMock.mockReturnValue({
      summary: "Provider request failed",
      hint: "Try another provider",
      detail: "Rate limited"
    })

    render(
      <PlaygroundMessage
        {...baseProps}
        onRegenerate={onRegenerate}
        onContinue={onContinue}
      />
    )

    expect(screen.getByRole("alert")).toHaveTextContent("Provider request failed")

    await user.click(screen.getByRole("button", { name: "Retry same model" }))
    expect(onRegenerate).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: "Switch model" }))
    expect(settingsEventSpy).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: "Try provider fallback" }))
    expect(updateChatModelSettingMock).toHaveBeenCalledWith("apiProvider", undefined)
    expect(antdMessageApi.info).toHaveBeenCalledWith(
      "Provider override cleared. Retrying with fallback routing."
    )
    expect(onRegenerate).toHaveBeenCalledTimes(2)

    await user.click(screen.getByRole("button", { name: "Continue from partial" }))
    expect(onContinue).toHaveBeenCalledTimes(1)

    window.removeEventListener(
      "tldw:open-model-settings",
      settingsEventSpy as EventListener
    )
  })

  it("renders interruption recovery status/actions for partial non-error responses", async () => {
    const user = userEvent.setup()
    const onRegenerate = vi.fn()
    const onContinue = vi.fn()

    render(
      <PlaygroundMessage
        {...baseProps}
        onRegenerate={onRegenerate}
        onContinue={onContinue}
        generationInfo={{
          interrupted: true,
          interruptionReason: "Stream ended early"
        }}
      />
    )

    const status = screen.getByRole("status")
    expect(status).toHaveTextContent("Generation was interrupted")
    expect(status).toHaveTextContent("Stream ended early")

    await user.click(screen.getByRole("button", { name: "Retry same model" }))
    expect(onRegenerate).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: "Continue from partial" }))
    expect(onContinue).toHaveBeenCalledTimes(1)
  })

  it("renders recovery for completed assistant turns with no visible text", async () => {
    const user = userEvent.setup()
    const onRegenerate = vi.fn()
    const settingsEventSpy = vi.fn()
    window.addEventListener(
      "tldw:open-model-settings",
      settingsEventSpy as EventListener
    )

    render(
      <PlaygroundMessage
        {...baseProps}
        message=""
        onRegenerate={onRegenerate}
      />
    )

    const status = screen.getByRole("status", {
      name: "Empty assistant response"
    })
    expect(status).toHaveTextContent("No response text was returned.")
    expect(status).toHaveTextContent(
      "The request completed, but the assistant did not return visible content."
    )

    await user.click(screen.getByRole("button", { name: "Retry same model" }))
    expect(onRegenerate).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: "Switch model" }))
    expect(settingsEventSpy).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole("button", { name: "Try provider fallback" }))
    expect(updateChatModelSettingMock).toHaveBeenCalledWith("apiProvider", undefined)
    expect(onRegenerate).toHaveBeenCalledTimes(2)

    window.removeEventListener(
      "tldw:open-model-settings",
      settingsEventSpy as EventListener
    )
  })

  it("renders a compact partial-save marker when stream transport drops after partial output", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{
          streamTransportInterrupted: true,
          partialResponseSaved: true,
          streamTransportInterruptionReason:
            "Could not establish connection. Receiving end does not exist."
        }}
      />
    )

    expect(
      screen.getByText("Connection dropped. Partial response saved.")
    ).toBeInTheDocument()
  })

  it("renders an image artifact event card with prompt and runtime metadata", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        message=""
        message_type={IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE}
        images={["data:image/png;base64,AAAA"]}
        generationInfo={{
          image_generation: {
            request: {
              prompt: "portrait of Lana smiling at golden hour",
              backend: "comfyui",
              width: 1024,
              height: 1024,
              format: "png",
              steps: 28,
              cfgScale: 6.5,
              sampler: "Euler a"
            },
            source: "generate-modal",
            refine: {
              model: "gpt-4o-mini",
              latencyMs: 842,
              diffStats: {
                baselineSegments: 4,
                candidateSegments: 5,
                sharedSegments: 3,
                overlapRatio: 0.6,
                addedCount: 2,
                removedCount: 1
              }
            },
            sync: {
              mode: "on",
              policy: "inherit",
              status: "synced"
            }
          }
        }}
      />
    )

    expect(screen.getByTestId("playground-image-event-card")).toBeInTheDocument()
    expect(screen.getByTestId("playground-image-event-prompt")).toHaveTextContent(
      "portrait of Lana smiling at golden hour"
    )
    expect(screen.getByText("Backend: comfyui")).toBeInTheDocument()
    expect(screen.getByText("Size: 1024x1024")).toBeInTheDocument()
    expect(screen.getByText("Steps: 28")).toBeInTheDocument()
    expect(screen.getByText("CFG: 6.5")).toBeInTheDocument()
    expect(screen.getByText("Generate menu")).toBeInTheDocument()
    expect(screen.getByText("Mirrored to server")).toBeInTheDocument()
    expect(
      screen.getByText("Refined with gpt-4o-mini (842 ms)")
    ).toBeInTheDocument()
  })

  it("does not render image artifact event metadata when generation payload is absent", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        message="regular assistant reply"
        message_type={IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE}
        generationInfo={{}}
      />
    )

    expect(
      screen.queryByTestId("playground-image-event-card")
    ).not.toBeInTheDocument()
  })

  it("renders image variant controls and compare preview for grouped image events", async () => {
    const user = userEvent.setup()
    const onSelectImageVariant = vi.fn()
    const onKeepImageVariant = vi.fn()
    const onDeleteImageVariant = vi.fn()
    const onDeleteAllImageVariants = vi.fn()

    render(
      <PlaygroundMessage
        {...baseProps}
        message=""
        messageId="image-assistant"
        message_type={IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE}
        images={["data:image/png;base64,BBBB"]}
        variants={[
          {
            id: "variant-1",
            message: "",
            images: ["data:image/png;base64,AAAA"]
          },
          {
            id: "variant-2",
            message: "",
            images: ["data:image/png;base64,BBBB"]
          }
        ]}
        activeVariantIndex={1}
        generationInfo={{
          image_generation: {
            request: {
              prompt: "portrait of Lana smiling at golden hour",
              backend: "comfyui"
            },
            source: "generate-modal"
          }
        }}
        onSelectImageVariant={onSelectImageVariant}
        onKeepImageVariant={onKeepImageVariant}
        onDeleteImageVariant={onDeleteImageVariant}
        onDeleteAllImageVariants={onDeleteAllImageVariants}
      />
    )

    expect(screen.getByTestId("playground-image-variant-strip")).toBeInTheDocument()

    await user.click(screen.getByTestId("playground-image-variant-select-0"))
    expect(onSelectImageVariant).toHaveBeenCalledWith({
      messageId: "image-assistant",
      variantIndex: 0
    })

    await user.click(screen.getByTestId("playground-image-variant-keep-active"))
    expect(onKeepImageVariant).toHaveBeenCalledWith({
      messageId: "image-assistant",
      variantIndex: 1
    })

    await user.click(screen.getByTestId("playground-image-variant-delete-0"))
    expect(onDeleteImageVariant).toHaveBeenCalledWith({
      messageId: "image-assistant",
      variantIndex: 0
    })

    await user.click(screen.getByTestId("playground-image-variant-compare-0"))
    expect(
      screen.getByTestId("playground-image-variant-compare-preview")
    ).toBeInTheDocument()

    await user.click(screen.getByTestId("playground-image-variant-delete-all"))
    expect(onDeleteAllImageVariants).toHaveBeenCalledWith({
      messageId: "image-assistant"
    })
  })
})
