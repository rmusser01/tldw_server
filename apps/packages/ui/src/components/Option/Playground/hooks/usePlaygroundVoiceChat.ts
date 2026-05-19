import React from "react"
import type { AudioCaptureRequestedSource } from "@/audio"
import type {
  DictationModePreference,
  DictationToggleIntent
} from "@/hooks/useDictationStrategy"
import type { SttSettings } from "@/hooks/useSttSettings"
import { useComposerVoiceChat } from "@/components/Chat/composer/hooks/useComposerVoiceChat"
import { withTemplateFallback } from "@/utils/template-guards"
import type { VoiceConversationAvailability } from "@/services/tldw/voice-conversation"

/**
 * Playground-specific voice orchestration. Layers two concerns on top of the
 * shared `useComposerVoiceChat` primitive:
 *   1. Voice-conversation back-and-forth — the "live voice chat" toggle, its
 *      stream state, status label, and a window-title side effect.
 *   2. The Playground tooltip text (with i18n keys + STT model details that
 *      are only meaningful here).
 *
 * Dictation orchestration (browser/server engines, source preferences,
 * diagnostics, toggle intents) is delegated to the shared hook. Sidepanel
 * uses the shared hook directly.
 *
 * Transcripts are written to the message field via Playground's
 * `setMessageValue(text, { collapseLarge: true, forceCollapse: true })` so
 * large pasted-then-redictated payloads stay collapsed in the textarea.
 */

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundVoiceChatDeps {
  /** Shared voice conversation availability contract */
  voiceConversationAvailability: VoiceConversationAvailability
  /** Voice chat enabled toggle */
  voiceChatEnabled: boolean
  setVoiceChatEnabled: (enabled: boolean) => void
  /** Voice chat stream hook instance */
  voiceChat: {
    state: string
    [key: string]: any
  }
  voiceChatMessages: {
    abandonTurn: () => void
    [key: string]: any
  }
  /** Server capabilities */
  canUseServerStt: boolean
  /** STT settings from storage */
  sttModel: string
  sttTemperature: number
  sttTask: string
  sttResponseFormat: string
  sttTimestampGranularities: string
  sttPrompt: string
  sttUseSegmentation: boolean
  sttSegK: number
  sttSegMinSegmentSize: number
  sttSegLambdaBalance: number
  sttSegUtteranceExpansionWidth: number
  sttSegEmbeddingsProvider: string
  sttSegEmbeddingsModel: string
  /** Dictation preferences from storage */
  dictationModeOverride: DictationModePreference | null
  dictationAutoFallbackEnabled: boolean
  autoStopTimeout: number
  autoSubmitVoiceMessage: boolean
  /** Language */
  speechToTextLanguage: string
  /** Callbacks */
  setMessageValue: (value: string, options?: any) => void
  submitForm: () => void
  /** Notification API */
  notificationApi: { error: (opts: any) => void; warning: (opts: any) => void }
  isSending: boolean
  isListening: boolean
  isServerDictating: boolean
  /** i18n */
  t: (key: string, ...args: any[]) => string
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePlaygroundVoiceChat(deps: UsePlaygroundVoiceChatDeps) {
  const {
    voiceConversationAvailability,
    voiceChatEnabled,
    setVoiceChatEnabled,
    voiceChat,
    voiceChatMessages,
    canUseServerStt,
    sttModel,
    sttTemperature,
    sttTask,
    sttResponseFormat,
    sttTimestampGranularities,
    sttPrompt,
    sttUseSegmentation,
    sttSegK,
    sttSegMinSegmentSize,
    sttSegLambdaBalance,
    sttSegUtteranceExpansionWidth,
    sttSegEmbeddingsProvider,
    sttSegEmbeddingsModel,
    dictationModeOverride,
    dictationAutoFallbackEnabled,
    autoStopTimeout,
    autoSubmitVoiceMessage,
    speechToTextLanguage,
    setMessageValue,
    submitForm,
    notificationApi,
    t
  } = deps

  const sttSettings = React.useMemo<SttSettings>(
    () => ({
      model: sttModel,
      temperature: sttTemperature,
      task: sttTask,
      responseFormat: sttResponseFormat,
      timestampGranularities: sttTimestampGranularities,
      prompt: sttPrompt,
      useSegmentation: sttUseSegmentation,
      segK: sttSegK,
      segMinSegmentSize: sttSegMinSegmentSize,
      segLambdaBalance: sttSegLambdaBalance,
      segUtteranceExpansionWidth: sttSegUtteranceExpansionWidth,
      segEmbeddingsProvider: sttSegEmbeddingsProvider,
      segEmbeddingsModel: sttSegEmbeddingsModel
    }),
    [
      sttModel,
      sttPrompt,
      sttResponseFormat,
      sttSegEmbeddingsModel,
      sttSegEmbeddingsProvider,
      sttSegK,
      sttSegLambdaBalance,
      sttSegMinSegmentSize,
      sttSegUtteranceExpansionWidth,
      sttTask,
      sttTemperature,
      sttTimestampGranularities,
      sttUseSegmentation
    ]
  )

  const handleTranscript = React.useCallback(
    (text: string) => {
      setMessageValue(text, { collapseLarge: true, forceCollapse: true })
    },
    [setMessageValue]
  )

  const handleAutoSubmit = React.useCallback(() => {
    submitForm()
  }, [submitForm])

  const composerVoice = useComposerVoiceChat({
    surface: "playground",
    canUseServerStt,
    speechToTextLanguage,
    sttSettings,
    dictationModeOverride,
    dictationAutoFallbackEnabled,
    autoStopTimeout,
    autoSubmitVoiceMessage,
    onTranscript: handleTranscript,
    onAutoSubmit: handleAutoSubmit
  })

  const {
    transcript,
    isListening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isServerDictating,
    startServerDictation,
    stopServerDictation,
    dictationAudioSourcePreference,
    setDictationAudioSourcePreference,
    dictationResolvedSourceKind,
    speechAvailable,
    speechUsesServer,
    dictationToggleIntent,
    dictationStrategy,
    startBrowserDictation,
    stopListening,
    handleDictationToggle
  } = composerVoice

  // --- Tooltip ---
  const speechTooltipText = React.useMemo(() => {
    if (!speechAvailable) {
      return t(
        "playground:actions.speechUnavailableBody",
        "Connect to a tldw server that exposes the audio transcriptions API to use dictation."
      ) as string
    }
    if (dictationStrategy.autoFallbackActive) {
      return t(
        "playground:tooltip.speechToTextBrowser",
        "Dictation via browser speech recognition"
      ) as string
    }
    if (speechUsesServer) {
      const sttModelLabel = sttModel || "whisper-1"
      const sttTaskLabel = sttTask === "translate" ? "translate" : "transcribe"
      const sttFormatLabel = (sttResponseFormat || "json").toUpperCase()
      const speechDetails = withTemplateFallback(
        t(
          "playground:tooltip.speechToTextDetails",
          "Uses {{model}} · {{task}} · {{format}}. Configure in Settings → General → Speech-to-Text.",
          {
            model: sttModelLabel,
            task: sttTaskLabel,
            format: sttFormatLabel
          } as any
        ),
        `Uses ${sttModelLabel} · ${sttTaskLabel} · ${sttFormatLabel}. Configure in Settings -> General -> Speech-to-Text.`
      )
      return (
        (t(
          "playground:tooltip.speechToTextServer",
          "Dictation via your tldw server"
        ) as string) +
        " " +
        speechDetails
      )
    }
    return t(
      "playground:tooltip.speechToTextBrowser",
      "Dictation via browser speech recognition"
    ) as string
  }, [
    dictationStrategy.autoFallbackActive,
    speechAvailable,
    speechUsesServer,
    sttModel,
    sttTask,
    sttResponseFormat,
    t
  ])

  // --- Voice chat status label ---
  const voiceChatStatusLabel = React.useMemo(() => {
    switch (voiceChat.state) {
      case "connecting":
        return t("playground:voiceChat.statusConnecting", "Connecting")
      case "listening":
        return t("playground:voiceChat.statusListening", "Listening")
      case "thinking":
        return t("playground:voiceChat.statusThinking", "Thinking")
      case "speaking":
        return t("playground:voiceChat.statusSpeaking", "Speaking")
      case "error":
        return t("playground:voiceChat.statusError", "Error")
      default:
        return t("playground:voiceChat.statusIdle", "Voice chat")
    }
  }, [t, voiceChat.state])

  // Update window title when voice chat is active
  React.useEffect(() => {
    if (!voiceChatEnabled || voiceChat.state === "idle") return
    const originalTitle = document.title
    const emoji = {
      connecting: "\u{1F50C}",
      listening: "\u{1F3A4}",
      thinking: "\u{1F4AD}",
      speaking: "\u{1F50A}",
      error: "\u26A0\uFE0F"
    }[voiceChat.state] || ""
    if (emoji) {
      document.title = `${emoji} ${voiceChatStatusLabel} - Chat`
    }
    return () => {
      document.title = originalTitle
    }
  }, [voiceChatEnabled, voiceChat.state, voiceChatStatusLabel])

  const voiceChatUnavailableMessage = React.useMemo(() => {
    const fallback = t(
      "playground:voiceChat.unavailableBody",
      "Connect to a tldw server with audio chat streaming enabled."
    )
    return voiceConversationAvailability.message
      ? t(voiceConversationAvailability.message, fallback)
      : fallback
  }, [t, voiceConversationAvailability.message])

  // --- Voice chat toggle ---
  const handleVoiceChatToggle = React.useCallback(() => {
    if (!voiceConversationAvailability.available) {
      notificationApi.error({
        message: t("playground:voiceChat.unavailableTitle", "Voice chat unavailable"),
        description: voiceChatUnavailableMessage
      })
      return
    }
    if (!voiceChatEnabled) {
      if (isListening) stopListening()
      if (isServerDictating) stopServerDictation()
      if (typeof window !== "undefined") {
        window.dispatchEvent(
          new CustomEvent("tldw:playground-starter-selected", {
            detail: { mode: "voice" }
          })
        )
      }
    }
    if (voiceChatEnabled) {
      voiceChatMessages.abandonTurn()
    }
    setVoiceChatEnabled(!voiceChatEnabled)
  }, [
    voiceChatEnabled,
    isListening,
    isServerDictating,
    notificationApi,
    setVoiceChatEnabled,
    stopListening,
    stopServerDictation,
    t,
    voiceChatUnavailableMessage,
    voiceChatMessages,
    voiceConversationAvailability
  ])

  return {
    // Speech recognition state
    transcript,
    isListening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    dictationAudioSourcePreference,
    dictationResolvedSourceKind,
    setDictationAudioSourcePreference,
    // Server dictation
    isServerDictating,
    startServerDictation,
    stopServerDictation,
    // Dictation strategy
    speechAvailable,
    speechUsesServer,
    dictationToggleIntent,
    // STT settings
    sttSettings,
    // Labels & tooltip
    voiceChatStatusLabel,
    speechTooltipText,
    // Handlers
    handleVoiceChatToggle,
    handleDictationToggle,
    startBrowserDictation,
    stopListening
  }
}

// Re-exported for callers that previously imported these via this module.
export type { AudioCaptureRequestedSource, DictationToggleIntent }
