import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useMessageState } from "@/components/Common/Playground/useMessageState"

const DEFAULT_KITTEN_MODEL = "KittenML/kitten-tts-nano-0.8"
const DEFAULT_KITTEN_VOICE = "Bella"

const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))

const ttsStatusState = vi.hoisted(() => ({
  calls: [] as Array<{ requireVoices?: boolean; tldwTtsModel?: string | null }>
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.values.has(key) ? storageState.values.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    cancel: vi.fn(),
    isSpeaking: false,
    speak: vi.fn(),
    ttsActionDisabled: false,
    ttsDisabledReason: null,
    ttsClipMeta: {}
  })
}))

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: (options: { requireVoices?: boolean; tldwTtsModel?: string }) => {
    ttsStatusState.calls.push(options)
    return {
      healthState: "healthy",
      voicesAvailable: true,
      audioHealthState: "healthy"
    }
  }
}))

vi.mock("@/hooks/useFeedback", () => ({
  useFeedback: () => ({
    thumb: null,
    detail: null,
    sourceFeedback: null,
    canSubmit: true,
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
      hasChatKnowledgeSave: true,
      hasNotes: true,
      hasFlashcards: true,
      hasChatDocuments: true,
      hasFeedbackExplicit: true,
      hasFeedbackImplicit: true
    }
  })
}))

vi.mock("@/hooks/useDiscoSkills", () => ({
  useDiscoSkills: () => ({
    enabled: false,
    stats: {},
    triggerProbabilityBase: 0,
    persistComments: false
  })
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
      apiProvider: undefined,
      updateSetting: vi.fn()
    })
}))

vi.mock("@/store", () => ({
  useStoreMessage: (selector: (state: any) => unknown) =>
    selector({ selectedModel: null })
}))

vi.mock("@/utils/chat-error-message", () => ({
  decodeChatErrorPayload: vi.fn(() => null)
}))

vi.mock("@/utils/chat-style", () => ({
  buildChatTextClass: vi.fn(() => "chat-text")
}))

vi.mock("../message-usage", () => ({
  resolveMessageCostUsd: vi.fn(() => null),
  resolveMessageUsage: vi.fn(() => ({ totalTokens: 0 }))
}))

vi.mock("../message-layout", () => ({
  resolveAvatarColumnAlignment: vi.fn(() => "items-start"),
  resolveMessageRenderSide: vi.fn(() => "right")
}))

vi.mock("@/utils/model-pricing", () => ({
  formatCost: vi.fn(() => "$0.00")
}))

vi.mock("../routing-fallback-audit", () => ({
  resolveFallbackAudit: vi.fn(() => null)
}))

vi.mock("@/utils/image-generation-chat", () => ({
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE: "image_generation",
  resolveImageGenerationMetadata: vi.fn(() => null)
}))

vi.mock("@/utils/character-mood", () => ({
  detectCharacterMood: vi.fn(() => ({ label: null })),
  normalizeCharacterMoodLabel: vi.fn(() => null),
  resolveCharacterBaseAvatarUrl: vi.fn(() => ""),
  resolveCharacterMoodImageUrl: vi.fn(() => "")
}))

describe("useMessageState TTS defaults", () => {
  beforeEach(() => {
    storageState.values.clear()
    ttsStatusState.calls = []
  })

  it("uses the canonical Kitten defaults on a clean profile", () => {
    const { result } = renderHook(() =>
      useMessageState({
        message: "hello",
        isBot: true,
        name: "Assistant",
        currentMessageIndex: 0,
        totalMessages: 1,
        isProcessing: false,
        isStreaming: false,
        conversationInstanceId: "conversation-1",
        onRegenerate: vi.fn(),
        onEditFormSubmit: vi.fn()
      })
    )

    expect(result.current.ttsProvider).toBe("tldw")
    expect(ttsStatusState.calls[0]).toEqual(
      expect.objectContaining({
        requireVoices: true,
        tldwTtsModel: DEFAULT_KITTEN_MODEL
      })
    )
  })
})
