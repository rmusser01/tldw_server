import { SidePanelBody } from "@/components/Sidepanel/Chat/body"
import { PlaygroundChat } from "@/components/Option/Playground/PlaygroundChat"
// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { PlaygroundMessage } from "../Message"
import { IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE } from "@/utils/image-generation-chat"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { normalizeMessageMetadataExtra } from "@/utils/dynamic-ui"

const storedPreferences = vi.hoisted(() => new Map<string, unknown>())
const detachedPresentation = vi.hoisted(() => ({ token: null as ({ originalReply?: { id?: string; message: string; isBot: boolean } }) | null, busy: false }))
const automaticActions = vi.hoisted(() => ({ speak: vi.fn(), copy: vi.fn(), skill: vi.fn(), skillsEnabled: false }))

afterEach(() => { detachedPresentation.token = null; detachedPresentation.busy = false; automaticActions.skillsEnabled = false; storedPreferences.clear() })

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
    t: (key: string, fallback?: string | { defaultValue?: string }, options?: Record<string, unknown>) => {
      const template = typeof fallback === "string" ? fallback : fallback?.defaultValue || key
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
    App: {
      useApp: () => ({ message: antdMessageApi })
    },
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
  useStorage: (key: string, defaultValue: unknown) => [storedPreferences.has(key) ? storedPreferences.get(key) : defaultValue, vi.fn()]
}))

vi.mock("@/components/Common/Markdown", () => ({
  default: ({ message }: { message: string }) => (
    <div data-testid="mock-markdown">{message}</div>
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
  MessageActionsBar: ({ onDelete }: { onDelete?: () => void }) => (
    <div data-testid="message-actions">
      {onDelete && <button onClick={onDelete}>Delete message</button>}
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
    speak: automaticActions.speak
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

const implicitFeedback = vi.hoisted(() => ({
  available: false,
  hook: vi.fn((_options: { enabled?: boolean }) => ({
    trackCopy: vi.fn(),
    trackSourcesExpanded: vi.fn(),
    trackSourceClick: vi.fn(),
    trackCitationUsed: vi.fn(),
    trackDwellTime: vi.fn()
  }))
}))

vi.mock("@/hooks/useImplicitFeedback", () => ({
  useImplicitFeedback: (options: { enabled?: boolean }) =>
    implicitFeedback.hook(options)
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasFeedbackExplicit: false,
      hasFeedbackImplicit: implicitFeedback.available
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
  useStoreMessageOption: Object.assign((
    selector: (state: {
      setReplyTarget: (...args: unknown[]) => void
      ragPinnedResults: unknown[]
      detachedHistoryResponse: object | null
      setMessages: (...args: unknown[]) => void
    }) => unknown
  ) =>
    selector({
      setReplyTarget: vi.fn(),
      ragPinnedResults: [],
      detachedHistoryResponse: detachedPresentation.token,
      setMessages: vi.fn()
    }), { getState: () => ({ streaming: detachedPresentation.busy, isProcessing: detachedPresentation.busy }) })
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
    enabled: automaticActions.skillsEnabled,
    stats: null,
    triggerProbabilityBase: 0,
    persistComments: false
  })
}))

vi.mock("@/utils/chat-error-message", () => ({
  decodeChatErrorPayload: decodeChatErrorPayloadMock
}))

vi.mock("@/utils/feedback", () => ({
  getSourceFeedbackKey: () => "feedback-key"
}))

vi.mock("@/utils/clipboard", () => ({
  copyToClipboard: automaticActions.copy
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
  attemptSkillTrigger: automaticActions.skill,
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
  generateID: () => crypto.randomUUID(),
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

describe("PlaygroundMessage implicit feedback in temporary mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    implicitFeedback.available = true
  })

  afterEach(() => {
    implicitFeedback.available = false
  })

  it.each([
    [true, false],
    [false, true]
  ])("temporaryChat=%s leaves passive feedback enabled=%s on a saved chat", (temporaryChat, enabled) => {
    render(
      <PlaygroundMessage
        {...baseProps}
        serverChatId="chat-1"
        serverMessageId="message-1"
        temporaryChat={temporaryChat}
      />
    )
    expect(implicitFeedback.hook).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled })
    )
  })
})

describe("PlaygroundMessage error recovery integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    decodeChatErrorPayloadMock.mockReturnValue(null)
  })

  it.each(["isStreaming", "isProcessing"] as const)("keeps the last saved reply passive during detached %s", async activeFlag => {
    detachedPresentation.token = {}
    const view = render(<PlaygroundMessage {...baseProps} {...{ [activeFlag]: true }} onStopStreaming={vi.fn()} />)
    expect(screen.getByTestId("chat-message")).not.toHaveAttribute("aria-busy")
    expect(screen.queryByTestId("chat-message-stop-streaming")).not.toBeInTheDocument()
    expect(screen.queryByTestId("playground-streaming-plain-text")).not.toBeInTheDocument()
    expect(await screen.findByTestId("mock-markdown")).toHaveTextContent("Sample assistant output")
    expect(screen.queryByText(/Generating response|Processing\.\.\./)).not.toBeInTheDocument()
    // The source row also stays passive if the timeline remounts while the request is held.
    view.unmount()
    const remounted = render(<PlaygroundMessage {...baseProps} {...{ [activeFlag]: true }} />)
    remounted.rerender(<PlaygroundMessage {...baseProps} />)
    detachedPresentation.token = null
    remounted.rerender(<PlaygroundMessage {...baseProps} />)
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
  })

  it("clears a previous response announcement when a detached request begins", () => {
    const view = render(<PlaygroundMessage {...baseProps} isStreaming />)
    view.rerender(<PlaygroundMessage {...baseProps} />)
    expect(screen.getByText("Response complete")).toBeInTheDocument()
    detachedPresentation.token = {}
    view.rerender(<PlaygroundMessage {...baseProps} isProcessing />)
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
    view.rerender(<PlaygroundMessage {...baseProps} />)
    detachedPresentation.token = null
    view.rerender(<PlaygroundMessage {...baseProps} />)
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
  })

  it("keeps a saved empty reply notice during detached generation", () => {
    detachedPresentation.token = {}
    render(<PlaygroundMessage {...baseProps} message="" metadataExtra={{ generation_status: "complete" }} isStreaming isProcessing />)
    expect(screen.getByText("No response text was returned.")).toBeInTheDocument()
    expect(screen.queryByText("Generating...")).not.toBeInTheDocument()
    expect(screen.queryByText("Generating response...")).not.toBeInTheDocument()
  })

  it("keeps a reasoning-only reply warning during detached generation", () => {
    detachedPresentation.token = {}
    render(<PlaygroundMessage {...baseProps} message="<think>Only reasoning</think>" isStreaming isProcessing />)
    expect(screen.getByRole("status")).toHaveTextContent("Interrupted")
    expect(screen.queryByText("Generating response...")).not.toBeInTheDocument()
  })

  it.each(["isStreaming", "isProcessing"] as const)("announces successful completion after %s ends", activeFlag => {
    const view = render(<PlaygroundMessage {...baseProps} {...{ [activeFlag]: true }} />)
    view.rerender(<PlaygroundMessage {...baseProps} />)
    expect(screen.getByText("Response complete")).toHaveAttribute("aria-live", "polite")
  })

  it.each(["isStreaming", "isProcessing"] as const)("announces the error without successful completion after %s fails", activeFlag => {
    const view = render(<PlaygroundMessage {...baseProps} {...{ [activeFlag]: true }} />)
    decodeChatErrorPayloadMock.mockReturnValue({
      summary: "Image support is not confirmed for this model.",
      hint: "Choose a model that supports images.",
      detail: ""
    })
    view.rerender(<PlaygroundMessage {...baseProps} message="failed response" />)
    expect(screen.getByRole("alert")).toHaveTextContent("Image support is not confirmed")
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
  })

  it("removes a pending completion announcement when the message becomes an error", () => {
    const view = render(<PlaygroundMessage {...baseProps} isStreaming />)
    view.rerender(<PlaygroundMessage {...baseProps} />)
    expect(screen.getByText("Response complete")).toBeInTheDocument()
    decodeChatErrorPayloadMock.mockReturnValue({
      summary: "Transport failed",
      hint: "Retry the request.",
      detail: ""
    })
    view.rerender(<PlaygroundMessage {...baseProps} message="failed response" />)
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
  })

  it.each(["isStreaming", "isProcessing"] as const)("does not resurrect completion when a failed response restarts %s", activeFlag => {
    const view = render(<PlaygroundMessage {...baseProps} {...{ [activeFlag]: true }} />)
    decodeChatErrorPayloadMock.mockReturnValue({
      summary: "Request failed",
      hint: "Retry the request.",
      detail: ""
    })
    view.rerender(<PlaygroundMessage {...baseProps} message="failed response" />)
    decodeChatErrorPayloadMock.mockReturnValue(null)
    view.rerender(<PlaygroundMessage {...baseProps} message="Retrying" {...{ [activeFlag]: true }} />)
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
  })

  it.each(["<think>Only reasoning</think>", "<think>Only reasoning"])("offers recovery for a settled restored reasoning-only row: %s", async message => {
    const retry = vi.fn()
    render(<PlaygroundMessage {...baseProps} message={message} onRegenerate={retry} />)
    expect(screen.getByText(/No final answer/)).toBeInTheDocument()
    expect(screen.getByText("Only reasoning")).toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: /Retry/ }))
    expect(retry).toHaveBeenCalledOnce()
  })

  it.each([
    { message: "<think>Working", isStreaming: true },
    { message: "<think>Working</think>Final answer" },
    { message: "<think>Working</think>", role: "user" as const, isBot: false },
    { message: "<think>Working</think>", toolCalls: [{ id: "tool", type: "function" as const, function: { name: "lookup", arguments: "{}" } }] }
  ])("does not label active or usable output as missing final: %j", overrides => {
    render(<PlaygroundMessage {...baseProps} {...overrides} />)
    expect(screen.queryByText(/No final answer/)).not.toBeInTheDocument()
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
    try {
      render(
        <PlaygroundMessage
          {...baseProps}
          message=""
          onRegenerate={onRegenerate}
        />
      )

      const status = screen.getByRole("status", {
        name: "No response text was returned."
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
    } finally {
      window.removeEventListener(
        "tldw:open-model-settings",
        settingsEventSpy as EventListener
      )
    }
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


it('opens a user editor from timeline navigation when portrait cards own rendering', () => {
  render(<PlaygroundMessage {...baseProps} isBot={false} role="user" messageId="timeline-user" message="Selected user row" />)
  act(() => window.dispatchEvent(new CustomEvent('tldw:edit-message', { detail: { messageId: 'timeline-user' } })))
  expect(screen.getByRole('textbox')).toBeInTheDocument()
})

for (const initialPortraits of [true, false]) {
  it.each(["save", "cancel"])(
    `keeps the active editor and draft while portrait preference changes from ${initialPortraits} until %s`,
    async (action) => {
      const user = userEvent.setup()
      storedPreferences.set("chatShowCharacterPortraits", initialPortraits)
      const submit = vi.fn()
      const props = {
        ...baseProps,
        isBot: false,
        role: "user" as const,
        messageId: "stable-user",
        message: "Original row",
        onEditFormSubmit: submit
      }
      const view = render(<PlaygroundMessage {...props} />)
      act(() =>
        window.dispatchEvent(
          new CustomEvent("tldw:edit-message", {
            detail: { messageId: "stable-user" }
          })
        )
      )
      const editor = screen.getByRole("textbox")
      await user.clear(editor)
      await user.type(editor, "Retained authored draft")
      storedPreferences.set("chatShowCharacterPortraits", !initialPortraits)
      view.rerender(<PlaygroundMessage {...props} />)
      expect(screen.getByRole("textbox")).toBe(editor)
      expect(editor).toHaveValue("Retained authored draft")
      await user.click(
        screen.getByRole("button", { name: action, exact: true })
      )
      if (action === "save")
        expect(submit).toHaveBeenCalledWith(
          "Retained authored draft",
          false,
          expect.any(Function),
          expect.any(Function)
        )
      else expect(submit).not.toHaveBeenCalled()
      expect(screen.queryByRole("textbox")).not.toBeInTheDocument()
      expect(screen.getByTestId("chat-message").tagName).toBe(
        initialPortraits ? "DIV" : "ARTICLE"
      )
      storedPreferences.clear()
    }
  )
}

it.each([true, false])(
  "keeps a same-text replacement target after an old edit acknowledges, portraits=%s",
  async (portraits) => {
    const user = userEvent.setup()
    storedPreferences.set("chatShowCharacterPortraits", portraits)
    storedPreferences.set("userChatBubble", true)
    let acknowledge!: () => void
    const persisted = new Promise<void>((resolve) => {
      acknowledge = resolve
    })
    void persisted.catch(() => undefined)
    const submit = vi.fn(() => persisted)
    const props = {
      ...baseProps,
      isBot: false,
      role: "user" as const,
      messageId: "target-A",
      message: "Same message",
      onEditFormSubmit: submit
    }
    try {
      const view = render(<PlaygroundMessage {...props} />)
      act(() =>
        window.dispatchEvent(
          new CustomEvent("tldw:edit-message", {
            detail: { messageId: "target-A" }
          })
        )
      )
      await user.clear(screen.getByRole("textbox"))
      await user.type(screen.getByRole("textbox"), "Older authored draft")
      fireEvent.click(screen.getByRole("button", { name: "save" }))
      expect(submit.mock.calls).toEqual([["Older authored draft", false]])
      expect(screen.getByRole("textbox")).toHaveValue("Older authored draft")

      view.rerender(<PlaygroundMessage {...props} messageId="target-B" />)
      expect(screen.getByRole("textbox")).toHaveValue("Same message")
      await user.clear(screen.getByRole("textbox"))
      await user.type(screen.getByRole("textbox"), "Replacement draft")
      await act(async () => acknowledge())
      expect(screen.getByRole("textbox")).toHaveValue("Replacement draft")
    } finally {
      await act(async () => {
        acknowledge()
        await persisted
      })
      storedPreferences.clear()
    }
  }
)

// Keep the real parent, timeline, message and edit form together for reconciliation.
const parentChat = vi.hoisted(() => ({
  messages: [] as Array<{
    id: string
    message: string
    name: string
    isBot: boolean
    role: string
  }>,
  historyId: "identity-chat",
  editMessage: vi.fn()
}))
vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => ({
    ...parentChat,
    temporaryChat: false,
    streaming: false,
    isProcessing: false,
    setMessages: vi.fn(),
    regenerateLastMessage: vi.fn()
  })
}))
vi.mock("@/store/webui", () => ({ useWebUI: () => ({ ttsEnabled: false }) }))
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: (options: {
    count: number
    getItemKey?: (index: number) => unknown
  }) => ({
    getVirtualItems: () =>
      Array.from({ length: options.count }, (_, index) => ({
        index,
        key: options.getItemKey?.(index) ?? index,
        start: index * 120
      })),
    getTotalSize: () => options.count * 120,
    measureElement: () => {},
    getOffsetForIndex: () => [0, "center"],
    scrollToOffset: vi.fn()
  })
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ...parentChat,
    compareMode: false,
    compareFeatureEnabled: false,
    streaming: false,
    isProcessing: false,
    setMessages: vi.fn(),
    onSubmit: vi.fn()
  })
}))
vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const queryClient = new actual.QueryClient()
  return {
    ...actual,
    useQueryClient: () => queryClient,
    useQuery: () => ({ data: [] })
  }
})
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => antdMessageApi
}))
vi.mock("@/hooks/chat/useDynamicUIActionBridge", () => ({
  useDynamicUIActionBridge: () => vi.fn()
}))

it.each([false, true])(
  "real parent preserves a moving editor through removal/reorder, bubble=%s",
  async (bubble) => {
    storedPreferences.set("chatShowCharacterPortraits", !bubble)
    storedPreferences.set("userChatBubble", true)
    parentChat.historyId = "identity-chat"
    const row = (id: string) => ({
      id,
      message: `Original ${id}`,
      name: "You",
      isBot: false,
      role: "user"
    })
    parentChat.messages = [row("before"), row("retained"), row("after")]
    const submitted: Array<{ id: string; value: string; send: boolean }> = []
    parentChat.editMessage
      .mockReset()
      .mockImplementation((index, value, _isUser, send) => {
        submitted.push({ id: parentChat.messages[index].id, value, send })
      })
    const view = render(<PlaygroundChat />)
    const user = userEvent.setup()
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:edit-message", {
          detail: { messageId: "retained" }
        })
      )
    )
    const editor = screen.getByRole("textbox")
    const form = editor.closest("form")
    await user.clear(editor)
    await user.type(editor, "retained unsaved draft")
    parentChat.messages = parentChat.messages.slice(1)
    view.rerender(<PlaygroundChat />)
    expect(screen.getByRole("textbox")).toBe(editor)
    expect(editor.closest("form")).toBe(form)
    expect(editor).toHaveValue("retained unsaved draft")
    parentChat.messages = [...parentChat.messages].reverse()
    view.rerender(<PlaygroundChat />)
    expect(screen.getByRole("textbox")).toBe(editor)
    await user.click(screen.getByRole("button", { name: "save" }))
    expect(submitted).toEqual([
      { id: "retained", value: "retained unsaved draft", send: false }
    ])
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:edit-message", {
          detail: { messageId: "retained" }
        })
      )
    )
    const cancelEditor = screen.getByRole("textbox")
    await user.clear(cancelEditor)
    await user.type(cancelEditor, "cancel draft")
    parentChat.messages = [...parentChat.messages].reverse()
    view.rerender(<PlaygroundChat />)
    expect(screen.getByRole("textbox")).toBe(cancelEditor)
    await user.click(
      screen.getByRole("button", { name: "cancel" })
    )
    expect(submitted).toHaveLength(1)
    expect(screen.queryByRole("textbox")).toBeNull()
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:edit-message", {
          detail: { messageId: "retained" }
        })
      )
    )
    await user.type(screen.getByRole("textbox"), " private old owner")
    parentChat.historyId = "replacement-owner"
    view.rerender(<PlaygroundChat />)
    expect(screen.queryByRole("textbox")).toBeNull()
    storedPreferences.clear()
  }
)


it("keeps nested reasoning tail out of the rendered final answer", async () => {
  render(
    <PlaygroundMessage
      {...baseProps}
      message="<think>outer <think>inner</think> tail</think>FINAL"
    />,
  );
  expect((await screen.findByTestId("mock-markdown")).textContent).toBe(
    "FINAL",
  );
});


it("keeps orphan reasoning closing markup out of the rendered final answer", async () => {
  render(
    <PlaygroundMessage
      {...baseProps}
      message="<think>brief check</think>FINAL</think>"
    />,
  );
  expect((await screen.findByTestId("mock-markdown")).textContent).toBe(
    "FINAL",
  );
});

it("offers honest recovery for only orphan reasoning markup", () => {
  render(<PlaygroundMessage {...baseProps} message="</think>" />);
  expect(screen.getByText(/No final answer/)).toBeInTheDocument();
});
it.each([false, true])(
  "real parent returns its held Save outcome and keeps a rejected draft, bubble=%s",
  async (bubble) => {
    storedPreferences.set("chatShowCharacterPortraits", !bubble)
    storedPreferences.set("userChatBubble", true)
    parentChat.historyId = "ack-chat"
    parentChat.messages = [
      {
        id: "ack-user",
        message: "Original row",
        name: "You",
        isBot: false,
        role: "user"
      }
    ]
    let reject!: (error: unknown) => void
    const held = new Promise<unknown>((_accept, refuse) => {
      reject = refuse
    })
    void held.catch(() => {})
    parentChat.editMessage.mockReset().mockReturnValue(held)
    render(<PlaygroundChat />)
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:edit-message", {
          detail: { messageId: "ack-user" }
        })
      )
    )
    const editor = screen.getByRole('textbox')
    fireEvent.change(editor, { target: { value: "Exact preserved edit  \n" } })
    fireEvent.click(screen.getByRole("button", { name: "save", exact: true }))
    try {
      expect(screen.queryByRole("textbox")).toBe(editor)
      expect(editor).toHaveValue("Exact preserved edit  \n")
    } finally {
      await act(async () => {
        reject(new Error("Current edit refused"))
        await held.catch(() => {})
      })
    }
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Your edited text is kept"
      )
    )
    expect(screen.getByRole("textbox")).toHaveValue("Exact preserved edit  \n")
    expect(parentChat.messages[0].message).toBe("Original row")
    parentChat.editMessage.mockResolvedValue({ status: "submitted" })
    fireEvent.click(screen.getByRole("button", { name: "save", exact: true }))
    await waitFor(() => expect(screen.queryByRole("textbox")).toBeNull())
    expect(parentChat.editMessage).toHaveBeenLastCalledWith(
      0,
      "Exact preserved edit  \n",
      true,
      false,
      expect.any(Function),
      expect.any(Function)
    )
    storedPreferences.clear()
  }
)

it.each([false, true])(
  "replacing a same-text message retires its old editor acknowledgement, bubble=%s",
  async (bubble) => {
    storedPreferences.set("chatShowCharacterPortraits", !bubble)
    storedPreferences.set("userChatBubble", true)
    let release!: (result: unknown) => void
    const held = new Promise<unknown>((resolve) => {
      release = resolve
    })
    const closeProps = {
      ...baseProps,
      isBot: false,
      role: "user" as const,
      message: "Same original text",
      messageId: "first",
      onEditFormSubmit: () => held
    }
    const view = render(<PlaygroundMessage {...closeProps} />)
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:edit-message", { detail: { messageId: "first" } })
      )
    )
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "First edited text" }
    })
    fireEvent.click(screen.getByRole("button", { name: "save", exact: true }))
    view.rerender(<PlaygroundMessage {...closeProps} messageId="second" />)
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:edit-message", {
          detail: { messageId: "second" }
        })
      )
    )
    expect(screen.getByRole("textbox")).toHaveValue("Same original text")
    await act(async () => {
      release({ status: "submitted" })
      await held
    })
    expect(screen.getByRole("textbox")).toHaveValue("Same original text")
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Second edited text" }
    })
    expect(screen.getByRole("textbox")).toHaveValue("Second edited text")
    storedPreferences.clear()
  }
)

it("the compact body returns its held Save outcome and retains a rejected draft", async () => {
  storedPreferences.set("chatShowCharacterPortraits", true)
  parentChat.historyId = "compact-ack-chat"
  parentChat.messages = [
    {
      id: "compact-ack-user",
      message: "Original compact row",
      name: "You",
      isBot: false,
      role: "user"
    }
  ]
  let reject!: (error: unknown) => void
  const held = new Promise<unknown>((_accept, refuse) => {
    reject = refuse
  })
  void held.catch(() => {})
  parentChat.editMessage.mockReset().mockReturnValue(held)
  render(<SidePanelBody />)
  act(() =>
    window.dispatchEvent(
      new CustomEvent("tldw:edit-message", {
        detail: { messageId: "compact-ack-user" }
      })
    )
  )
  const editor = screen.getByRole("textbox")
  fireEvent.change(editor, { target: { value: "Exact compact edit  \n" } })
  fireEvent.click(screen.getByRole("button", { name: "save", exact: true }))
  try {
    expect(screen.queryByRole("textbox")).toBe(editor)
    expect(editor).toHaveValue("Exact compact edit  \n")
  } finally {
    await act(async () => {
      reject(new Error("Compact edit refused"))
      await held.catch(() => {})
    })
  }
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Your edited text is kept"
  )
  expect(screen.getByRole("textbox")).toHaveValue("Exact compact edit  \n")
  expect(parentChat.messages[0].message).toBe("Original compact row")
  storedPreferences.clear()
})


describe("PlaygroundMessage recorded response presentation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    decodeChatErrorPayloadMock.mockReturnValue(null)
  })

  it("shows the saved response model and usage instead of current or legacy values", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{
          model: "legacy-model",
          usage: { total_tokens: 99 },
          cost_usd: 99
        }}
        metadataExtra={{
          model_id: "saved-model",
          provider: "saved-provider",
          generation_status: "complete",
          finish_reason: "stop",
          usage: { prompt_tokens: 11, completion_tokens: 2, total_tokens: 13 }
        }}
      />
    )
    expect(screen.getByTestId("message-model-badge")).toHaveTextContent(
      "saved-model"
    )
    expect(screen.getByTestId("message-model-badge")).toHaveAttribute(
      "title",
      "saved-provider · saved-model"
    )
    expect(screen.getByText(/13.*tok/)).toBeInTheDocument()
    expect(screen.queryByText(/99.*tok/)).not.toBeInTheDocument()
    expect(screen.queryByText(/\$99/)).not.toBeInTheDocument()
  })

  it("does not pair a recorded provider with an unrecorded legacy model or cost", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{
          model: "legacy-model",
          usage: { total_tokens: 99 },
          cost_usd: 99
        }}
        metadataExtra={{
          generation_status: "complete",
          provider: "saved-provider"
        }}
      />
    )
    expect(screen.getByTestId("message-model-badge")).toHaveAttribute(
      "title",
      "legacy-model"
    )
    expect(
      screen.queryByText(/saved-provider.*legacy-model|\$99/)
    ).not.toBeInTheDocument()
  })

  it.each([{ model_id: "saved-model" }, { provider: "saved-provider" }])(
    "does not attach unrecorded legacy usage to recorded provenance %o",
    (metadata) => {
      render(
        <PlaygroundMessage
          {...baseProps}
          generationInfo={{
            model: "legacy-model",
            usage: { total_tokens: 99 }
          }}
          metadataExtra={{ generation_status: "complete", ...metadata }}
        />
      )
      expect(screen.queryByText(/99.*tok/)).not.toBeInTheDocument()
    }
  )

  it.each([
    ["interrupted", "Interrupted"],
    ["stopped", "Stopped"],
    ["length", "Output limit reached"],
    ["error", "Generation failed"]
  ])(
    "shows saved %s status without granting local recovery actions",
    (status, label) => {
      render(
        <PlaygroundMessage
          {...baseProps}
          metadataExtra={{
            generation_status: status,
            model_id: "saved-model",
            interrupted: true,
            stopped: true,
            canRetry: true,
            local_history: { generationInfo: { interrupted: true } }
          }}
        />
      )
      expect(screen.getByRole("status")).toHaveTextContent(label)
      expect(
        screen.queryByRole("button", { name: "Retry same model" })
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Continue from partial" })
      ).not.toBeInTheDocument()
    }
  )

  it.each(["", "<think>Unfinished reasoning</think>"])(
    "keeps a recorded partial passive with content %s",
    (message) => {
      render(
        <PlaygroundMessage
          {...baseProps}
          message={message}
          metadataExtra={{ generation_status: "interrupted" }}
        />
      )
      expect(screen.getByRole("status")).toHaveTextContent("Interrupted")
      expect(
        screen.queryByRole("button", { name: "Retry same model" })
      ).not.toBeInTheDocument()
    }
  )

  it("does not announce successful completion when the saved stream ended partially", () => {
    const view = render(<PlaygroundMessage {...baseProps} isStreaming />)
    view.rerender(
      <PlaygroundMessage
        {...baseProps}
        metadataExtra={{ generation_status: "stopped" }}
      />
    )
    expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent("Stopped")
  })

  it("keeps recorded zero usage instead of falling back to old usage", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{ usage: { total_tokens: 99 } }}
        metadataExtra={{
          generation_status: "complete",
          usage: { total_tokens: 0 }
        }}
      />
    )
    expect(screen.getByText("0 tokens")).toBeInTheDocument()
    expect(screen.queryByText(/99.*tok/)).not.toBeInTheDocument()
  })

  it.each([
    [{ total_tokens: 13 }, "13 tokens"],
    [{ prompt_tokens: 11 }, "11 prompt"]
  ])("shows only reported saved usage fields %o", (usage, label) => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{
          usage: { prompt_tokens: 99, completion_tokens: 99, total_tokens: 198 }
        }}
        metadataExtra={{ generation_status: "complete", usage }}
      />
    )
    expect(screen.getByText(label as string)).toBeInTheDocument()
    expect(
      screen.queryByText(/0 completion|99 prompt|198 tokens/)
    ).not.toBeInTheDocument()
  })

  it.each([
    { isProcessing: true, isStreaming: false },
    { isProcessing: false, isStreaming: true }
  ])("keeps saved status passive while another turn is busy %o", (busy) => {
    const savedProps = {
      ...baseProps,
      currentMessageIndex: 0,
      totalMessages: 3,
      generationInfo: { interrupted: true, stopped: true },
      metadataExtra: { generation_status: "stopped" }
    }
    const view = render(<PlaygroundMessage {...savedProps} />)
    view.rerender(<PlaygroundMessage {...savedProps} {...busy} />)
    expect(screen.getByRole("status")).toHaveTextContent("Stopped")
    expect(
      screen.queryByRole("button", { name: "Retry same model" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Continue from partial" })
    ).not.toBeInTheDocument()
  })

  it("does not attribute a saved stop to the current user", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        metadataExtra={{ generation_status: "stopped" }}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(
      "This reply was stopped. The saved content is kept."
    )
    expect(
      screen.queryByText(
        "You stopped this reply. What it wrote so far is kept."
      )
    ).not.toBeInTheDocument()
  })

  it.each(["Saved answer", "", "<think>Reasoning only</think>"])(
    "keeps a recorded complete response from inheriting private recovery with content %s",
    (message) => {
      render(
        <PlaygroundMessage
          {...baseProps}
          message={message}
          generationInfo={{ interrupted: true, stopped: true }}
          metadataExtra={{ generation_status: "complete" }}
        />
      )
      expect(screen.queryByText("Interrupted")).not.toBeInTheDocument()
      expect(screen.queryByText("Stopped")).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Retry same model" })
      ).not.toBeInTheDocument()
    }
  )

  it("keeps typed image-event presentation separate from text-generation metadata", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        message_type={IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE}
        generationInfo={{ model: "image-model" }}
        metadataExtra={{
          model_id: "text-model",
          generation_status: "interrupted"
        }}
      />
    )
    expect(screen.getByTestId("message-model-badge")).toHaveTextContent(
      "image-model"
    )
    expect(screen.queryByText("Interrupted")).not.toBeInTheDocument()
  })
})

it("reads recorded response provenance after the real server fallback mapper", async () => {
  decodeChatErrorPayloadMock.mockReturnValue(null)
  const { mapServerChatMessagesToPlaygroundMessages } = await vi.importActual<
    typeof import("@/hooks/chat/useServerChatLoader")
  >("@/hooks/chat/useServerChatLoader")
  const [message] = mapServerChatMessagesToPlaygroundMessages({
    assistantName: "Mira",
    characterId: "4",
    serverMessages: [
      {
        id: "recorded-reply",
        role: "assistant",
        content: "Saved partial response.",
        created_at: "2026-10-07T12:00:00Z",
        metadata_extra: {
          generation_status: "length",
          model_id: "saved-model",
          provider: "openai"
        }
      }
    ]
  })
  render(<PlaygroundMessage {...baseProps} {...message} />)
  expect(screen.getByTestId("message-model-badge")).toHaveTextContent(
    "saved-model"
  )
  expect(screen.getByRole("status")).toHaveTextContent("Output limit reached")
  expect(
    screen.queryByRole("button", { name: "Retry same model" })
  ).not.toBeInTheDocument()
})

it("reads captured native metadata without importing private local recovery fields", async () => {
  decodeChatErrorPayloadMock.mockReturnValue(null)
  const { formatSelectedHistory } =
    await vi.importActual<typeof import("@/db/dexie/helpers")>(
      "@/db/dexie/helpers"
    )
  const nodes = [
    { id: "u", revision: "1", role: "user", parent_id: null, settled: true },
    { id: "a", revision: "1", role: "assistant", parent_id: "u", settled: true }
  ]
  const capture: import("@/types/history-selection").HistorySelectionCaptureV1 =
    {
      status: "captured",
      purpose: "send",
      storage_context_digest: "storage",
      rows: nodes,
      snapshot: {
        version: 1,
        owner_key: "native-owner",
        conversation_id: "chat",
        nodes,
        fences: { conversation: "c", history: "h", settings: "s" },
        source_digest: "source",
        storage_context_digest: "storage",
        interpretation_status: { kind: "parent_graph_v1" }
      },
      view: {
        view_session_id: "view",
        owner_key: "native-owner",
        conversation_id: "chat",
        interpretation: { kind: "parent_graph_v1" },
        cursor: { kind: "after_message", message_id: "a" },
        selection_revision: 1
      },
      selected_content: [
        { id: "u", revision: "1", message: "Question", images: [] },
        {
          id: "a",
          revision: "1",
          message: "Saved partial.",
          images: [],
          extra_metadata: {
            generation_status: "interrupted",
            model_id: "saved-model",
            provider: "openai",
            local_history: {
              generationInfo: { interrupted: true, stopped: true },
              modelId: "private-model"
            }
          }
        }
      ]
    }
  const before = structuredClone(capture)
  const message = formatSelectedHistory(capture).messages[1]
  render(<PlaygroundMessage {...baseProps} {...message} />)
  expect(screen.getByTestId("message-model-badge")).toHaveTextContent(
    "saved-model"
  )
  expect(screen.getByRole("status")).toHaveTextContent("Interrupted")
  expect(
    screen.queryByRole("button", { name: "Retry same model" })
  ).not.toBeInTheDocument()
  expect(capture).toEqual(before)
})

it("reads recorded response metadata after the actual compact server mapper", () => {
  decodeChatErrorPayloadMock.mockReturnValue(null)
  // Execute the private mapper source, with its real normalization dependencies.
  // Existing mounted compact suites qualify loading/tab ownership separately.
  const route = ts.createSourceFile(
    "sidepanel-chat.tsx",
    readFileSync(
      path.resolve(__dirname, "../../../../routes/sidepanel-chat.tsx"),
      "utf8"
    ),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX
  )
  const names = ["normalizeServerChatMessageId", "mapServerChatMessages"]
  const declarations = route.statements
    .filter(ts.isVariableStatement)
    .filter((statement) =>
      statement.declarationList.declarations.some((declaration) =>
        names.includes(declaration.name.getText(route))
      )
    )
  expect(declarations).toHaveLength(2)
  const compiled = ts.transpileModule(
    declarations.map((declaration) => declaration.getText(route)).join("\n") +
      "\nreturn mapServerChatMessages",
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
  ).outputText
  const map = new Function(
    "normalizeChatRole",
    "normalizeMessageMetadataExtra",
    compiled
  )(normalizeChatRole, normalizeMessageMetadataExtra)
  const source = [
    {
      id: "compact-reply",
      role: "assistant",
      content: "Saved compact partial.",
      created_at: "2026-10-07T12:00:00Z",
      metadata_extra: {
        model_id: "compact-model",
        provider: "openai",
        generation_status: "stopped"
      }
    }
  ]
  const before = structuredClone(source)
  const result = map(source)
  render(<PlaygroundMessage {...baseProps} {...result.mappedMessages[0]} />)
  expect(screen.getByTestId("message-model-badge")).toHaveTextContent(
    "compact-model"
  )
  expect(screen.getByRole("status")).toHaveTextContent("Stopped")
  expect(
    screen.queryByRole("button", { name: "Retry same model" })
  ).not.toBeInTheDocument()
  expect(result.history).toEqual([
    { role: "assistant", content: "Saved compact partial." }
  ])
  expect(source).toEqual(before)
})


describe("saved reply recovery during detached work", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })
  it.each(["", "<think>Only reasoning</think>"])(
    "keeps the old notice without mutating recovery actions for %s",
    (text) => {
      detachedPresentation.token = {}
      render(
        <PlaygroundMessage
          {...baseProps}
          message={text}
          isStreaming
          isProcessing
        />
      )
      expect(
        screen.queryByRole("button", { name: "Retry same model" })
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Try provider fallback" })
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: "Switch model" })
      ).not.toBeInTheDocument()
      expect(updateChatModelSettingMock).not.toHaveBeenCalled()
    }
  )

  it("a retained provider fallback cannot mutate or retry before the new busy render", () => {
    const onRegenerate = vi.fn()
    const view = render(
      <PlaygroundMessage
        {...baseProps}
        message="<think>Only reasoning</think>"
        onRegenerate={onRegenerate}
      />
    )
    const action = screen.getByRole("button", { name: "Try provider fallback" })
    detachedPresentation.token = {}
    detachedPresentation.busy = true
    fireEvent.click(action)
    expect(updateChatModelSettingMock).not.toHaveBeenCalled()
    expect(onRegenerate).not.toHaveBeenCalled()
    view.rerender(
      <PlaygroundMessage
        {...baseProps}
        message="<think>Only reasoning</think>"
        onRegenerate={onRegenerate}
        isStreaming
      />
    )
    expect(
      screen.queryByRole("button", { name: "Try provider fallback" })
    ).not.toBeInTheDocument()
  })

  it("a non-followed detached reply never replays automatic actions on its unchanged source", async () => {
    storedPreferences.set("autoCopyResponseToClipboard", true)
    storedPreferences.set("isTTSAutoPlayEnabled", true)
    automaticActions.skillsEnabled = true
    detachedPresentation.token = {
      originalReply: { message: baseProps.message, isBot: true }
    }
    const view = render(
      <PlaygroundMessage {...baseProps} isTTSEnabled isStreaming isProcessing />
    )
    view.rerender(<PlaygroundMessage {...baseProps} isTTSEnabled />)
    detachedPresentation.token = null
    view.rerender(<PlaygroundMessage {...baseProps} isTTSEnabled />)
    await act(async () => {
      await Promise.resolve()
    })
    expect(automaticActions.copy).not.toHaveBeenCalled()
    expect(automaticActions.speak).not.toHaveBeenCalled()
    expect(automaticActions.skill).not.toHaveBeenCalled()
  })
})


it("a newly installed edited reply can run automatic actions once", async () => {
  vi.clearAllMocks()
  storedPreferences.set("autoCopyResponseToClipboard", true)
  storedPreferences.set("isTTSAutoPlayEnabled", true)
  automaticActions.skillsEnabled = true
  detachedPresentation.token = {
    originalReply: { id: "source", message: baseProps.message, isBot: true }
  }
  const view = render(
    <PlaygroundMessage
      {...baseProps}
      messageId="source"
      isTTSEnabled
      isStreaming
      isProcessing
    />
  )
  view.rerender(
    <PlaygroundMessage
      {...baseProps}
      messageId="new-result"
      message="Edited answer"
      isTTSEnabled
    />
  )
  detachedPresentation.token = null
  view.rerender(
    <PlaygroundMessage
      {...baseProps}
      messageId="new-result"
      message="Edited answer"
      isTTSEnabled
    />
  )
  await waitFor(() => {
    expect(automaticActions.copy).toHaveBeenCalledOnce()
    expect(automaticActions.speak).toHaveBeenCalledOnce()
    expect(automaticActions.skill).toHaveBeenCalledOnce()
  })
})


it.each([true, false, undefined])(
  "awaited Delete confirmation publishes success only for an accepted outcome (%s)",
  async (outcome) => {
    vi.clearAllMocks()
    const { Modal } = await import("antd")
    let release!: (value: boolean | undefined) => void
    const remove = vi.fn(
      () =>
        new Promise<boolean | undefined>((resolve) => {
          release = resolve
        })
    )
    render(<PlaygroundMessage {...baseProps} onDeleteMessage={remove} />)
    fireEvent.click(screen.getByRole("button", { name: "Delete message" }))
    const confirmation = vi.mocked(Modal.confirm).mock.calls.at(-1)![0]
    let pending!: Promise<unknown>
    act(() => {
      pending = confirmation.onOk!() as Promise<unknown>
    })
    expect(remove).toHaveBeenCalledOnce()
    expect(antdMessageApi.success).not.toHaveBeenCalled()
    await act(async () => {
      release(outcome)
      await pending
    })
    expect(antdMessageApi.success).toHaveBeenCalledTimes(
      outcome === false ? 0 : 1
    )
  }
)

it("Delete failure uses recovery copy and never announces success", async () => {
  vi.clearAllMocks()
  const { Modal } = await import("antd")
  render(
    <PlaygroundMessage
      {...baseProps}
      onDeleteMessage={async () => {
        throw new Error("message_has_descendants")
      }}
    />
  )
  fireEvent.click(screen.getByRole("button", { name: "Delete message" }))
  const confirmation = vi.mocked(Modal.confirm).mock.calls.at(-1)![0]
  await act(async () => {
    await confirmation.onOk!()
  })
  expect(antdMessageApi.success).not.toHaveBeenCalled()
  expect(antdMessageApi.error).toHaveBeenCalledWith(
    expect.stringMatching(/replies.*first/i)
  )
})


describe("preserved reply before recovery", () => {
  it.each(["interrupted", "empty", "provider error"] as const)("unsupported persona disables %s generation recovery before provider settings writes", state => {
    if (state === "provider error") decodeChatErrorPayloadMock.mockReturnValue({ summary: "Provider failed" })
    const onRegenerate = vi.fn(), onContinue = vi.fn()
    const reason = "Use a persona prompt overlay or choose a character for tracked chat."
    const props = { ...baseProps, onRegenerate, onContinue,
      message: state === "empty" ? "" : baseProps.message,
      generationInfo: state === "interrupted" ? { interrupted: true } : undefined }
    const view = render(<PlaygroundMessage {...props} regenerateDisabledReason={reason} />)
    for (const label of ["Retry same model", "Try provider fallback", ...(state !== "empty" ? ["Continue from partial"] : [])]) {
      const action = screen.getByRole("button", { name: label })
      expect(action).toBeDisabled()
      expect(action).toHaveAttribute("title", reason)
      fireEvent.click(action)
    }
    expect(screen.getByRole("button", { name: "Switch model" })).toBeEnabled()
    expect(updateChatModelSettingMock).not.toHaveBeenCalled()
    expect(onRegenerate).not.toHaveBeenCalled()
    expect(onContinue).not.toHaveBeenCalled()
    view.rerender(<PlaygroundMessage {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "Try provider fallback" }))
    expect(updateChatModelSettingMock).toHaveBeenCalledWith("apiProvider", undefined)
    expect(onRegenerate).toHaveBeenCalledOnce()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    decodeChatErrorPayloadMock.mockReturnValue(null)
  })
  afterEach(() => {
    parentChat.messages = []
    parentChat.historyId = "identity-chat"
  })
  const cases: Array<{
    name: string
    props: Partial<React.ComponentProps<typeof PlaygroundMessage>>
  }> = [
    {
      name: "local deliberate stop",
      props: { generationInfo: { interrupted: true, stopped: true } }
    },
    {
      name: "local interruption",
      props: {
        generationInfo: {
          interrupted: true,
          interruptionReason: "Stream ended early"
        }
      }
    },
    {
      name: "saved stopped reply",
      props: { metadataExtra: { generation_status: "stopped" } }
    },
    {
      name: "saved interrupted reply",
      props: { metadataExtra: { generation_status: "interrupted" } }
    },
    {
      name: "saved transport partial",
      props: {
        generationInfo: {
          streamTransportInterrupted: true,
          partialResponseSaved: true
        }
      }
    }
  ]
  it.each(
    cases.flatMap(({ name, props }) =>
      (["isStreaming", "isProcessing"] as const).map((activeFlag) => ({
        name,
        props,
        activeFlag
      }))
    )
  )(
    "$name announces its partial outcome after $activeFlag instead of completion",
    ({ props, activeFlag }) => {
      const view = render(
        <PlaygroundMessage {...baseProps} {...{ [activeFlag]: true }} />
      )
      view.rerender(<PlaygroundMessage {...baseProps} {...props} />)
      expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite")
      expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
    }
  )
  it("announces recorded completion over stale transient partial flags", () => {
    const view = render(<PlaygroundMessage {...baseProps} isStreaming />)
    view.rerender(
      <PlaygroundMessage
        {...baseProps}
        metadataExtra={{ generation_status: "complete" }}
        generationInfo={{
          interrupted: true,
          stopped: true,
          streamTransportInterrupted: true,
          partialResponseSaved: true
        }}
      />
    )
    expect(screen.getByText("Response complete")).toHaveAttribute(
      "aria-live",
      "polite"
    )
    expect(screen.queryByText("Stopped")).not.toBeInTheDocument()
  })
  it.each(cases)(
    "$name keeps prose, reasoning and the received image before the notice",
    async ({ props }) => {
      render(
        <PlaygroundMessage
          {...baseProps}
          message="<think>Kept reasoning</think>Kept partial answer"
          images={["data:image/png;base64,AAAA"]}
          {...props}
        />
      )
      const answer = await screen.findByText("Kept partial answer"),
        reason = screen.getByText("Kept reasoning"),
        image = screen.getByRole("img", { name: "Uploaded Image" }),
        notice = screen.getByRole("status")
      for (const content of [answer, reason, image])
        expect(
          content.compareDocumentPosition(notice) &
            Node.DOCUMENT_POSITION_FOLLOWING
        ).toBeTruthy()
    }
  )
  it.each(["full", "compact"] as const)(
    "the real %s parent keeps a stopped reply before recovery without provider fallback",
    async (surface) => {
      const reply = {
        id: "kept-reply",
        message: "Kept parent answer",
        name: "Assistant",
        isBot: true,
        role: "assistant",
        generationInfo: { interrupted: true, stopped: true }
      }
      parentChat.messages = [reply]
      render(surface === "full" ? <PlaygroundChat /> : <SidePanelBody />)
      const content = await screen.findByText(reply.message),
        notice = screen.getByRole("status")
      expect(
        content.compareDocumentPosition(notice) &
          Node.DOCUMENT_POSITION_FOLLOWING
      ).toBeTruthy()
      expect(
        screen.queryByRole("button", { name: "Try provider fallback" })
      ).toBeNull()
      expect(updateChatModelSettingMock).not.toHaveBeenCalled()
    }
  )
  it("a deliberate stop offers each real recovery action once and never mutates routing by itself", () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        generationInfo={{ interrupted: true, stopped: true }}
      />
    )
    expect(
      screen.queryByRole("button", { name: "Try provider fallback" })
    ).toBeNull()
    const notice = screen.getByRole("status")
    for (const label of [
      "Retry same model",
      "Switch model",
      "Continue from partial"
    ]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument()
      expect(notice.textContent!.split(label)).toHaveLength(2)
    }
    expect(notice).not.toHaveTextContent("Recommended next actions")
    expect(updateChatModelSettingMock).not.toHaveBeenCalled()
    expect(baseProps.onRegenerate).not.toHaveBeenCalled()
  })
  it.each(["interrupted", "empty", "provider error"] as const)(
    "$0 offers provider fallback but lists actual %s actions once",
    (state) => {
      if (state === "provider error")
        decodeChatErrorPayloadMock.mockReturnValue({
          summary: "Provider failed"
        })
      render(
        <PlaygroundMessage
          {...baseProps}
          message={state === "empty" ? "" : baseProps.message}
          generationInfo={
            state === "interrupted" ? { interrupted: true } : undefined
          }
        />
      )
      expect(
        screen.getByRole("button", { name: "Try provider fallback" })
      ).toBeInTheDocument()
      const notice = screen.getByRole(
        state === "provider error" ? "alert" : "status"
      )
      expect(notice).not.toHaveTextContent("Recommended next actions")
      expect(notice.textContent!.split("Try provider fallback")).toHaveLength(2)
      if (state === "interrupted")
        expect(notice).not.toHaveTextContent(
          "You can retry, switch model, or continue"
        )
      expect(updateChatModelSettingMock).not.toHaveBeenCalled()
    }
  )
  it("reasoning-only recovery states the missing answer without promising absent Continue or repeating Retry", async () => {
    render(
      <PlaygroundMessage
        {...baseProps}
        message="<think>Kept reasoning</think>"
        onContinue={undefined}
        generationInfo={{
          interruptionReason:
            "No final answer was generated. Retry or continue to request an answer."
        }}
      />
    )
    await screen.findByText("Kept reasoning")
    expect(screen.getByRole("status")).toHaveTextContent(
      "No final answer was generated."
    )
    expect(screen.getByRole("status")).not.toHaveTextContent(
      "Retry or continue to request an answer"
    )
    expect(
      screen.queryByRole("button", { name: "Continue from partial" })
    ).toBeNull()
  })
})
