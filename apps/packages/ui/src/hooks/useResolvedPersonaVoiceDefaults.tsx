import React from "react"
import { useStorage } from "@plasmohq/storage/hook"

import { useSttSettings } from "@/hooks/useSttSettings"
import { useVoiceChatSettings } from "@/hooks/useVoiceChatSettings"
import {
  DEFAULT_TLDW_TTS_VOICE,
  DEFAULT_TTS_PROVIDER
} from "@/services/tts"

export type PersonaConfirmationMode =
  | "always"
  | "destructive_only"
  | "never"

export type PersonaWakeBehavior =
  | "one_shot"
  | "continuous"
  | "push_to_talk_after_wake"

export type PersonaVoiceDefaults = {
  stt_language?: string | null
  stt_model?: string | null
  tts_provider?: string | null
  tts_voice?: string | null
  confirmation_mode?: PersonaConfirmationMode | null
  wake_behavior?: PersonaWakeBehavior | null
  voice_chat_trigger_phrases?: string[]
  auto_resume?: boolean | null
  barge_in?: boolean | null
  auto_commit_enabled?: boolean | null
  vad_threshold?: number | null
  min_silence_ms?: number | null
  turn_stop_secs?: number | null
  min_utterance_secs?: number | null
}

export type ResolvedPersonaVoiceDefaults = {
  sttLanguage: string
  sttModel: string
  ttsProvider: string
  ttsVoice: string
  confirmationMode: PersonaConfirmationMode
  wakeBehavior: PersonaWakeBehavior
  voiceChatTriggerPhrases: string[]
  autoResume: boolean
  bargeIn: boolean
  autoCommitEnabled: boolean
  vadThreshold: number
  minSilenceMs: number
  turnStopSecs: number
  minUtteranceSecs: number
}

const DEFAULT_STT_LANGUAGE = "en-US"
const DEFAULT_OPENAI_VOICE = "alloy"
const DEFAULT_CONFIRMATION_MODE: PersonaConfirmationMode = "destructive_only"
const DEFAULT_WAKE_BEHAVIOR: PersonaWakeBehavior = "one_shot"
export const PERSONA_TURN_DETECTION_BALANCED_DEFAULTS = {
  autoCommitEnabled: true,
  vadThreshold: 0.5,
  minSilenceMs: 250,
  turnStopSecs: 0.2,
  minUtteranceSecs: 0.4
} as const

const normalizeText = (value: string | null | undefined): string | null => {
  const trimmed = String(value || "").trim()
  return trimmed ? trimmed : null
}

const normalizePhrases = (value: string[] | null | undefined): string[] => {
  const seen = new Set<string>()
  const next: string[] = []
  for (const item of value || []) {
    const trimmed = String(item || "").trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    next.push(trimmed)
  }
  return next
}

const normalizeWakeBehavior = (
  value: PersonaWakeBehavior | null | undefined
): PersonaWakeBehavior =>
  value === "continuous" ||
  value === "push_to_talk_after_wake" ||
  value === "one_shot"
    ? value
    : DEFAULT_WAKE_BEHAVIOR

const resolveDefaultTtsVoice = (
  provider: string,
  tldwVoice: string,
  openAiVoice: string,
  elevenLabsVoice: string
): string => {
  const normalizedProvider = String(provider || "").trim().toLowerCase()
  if (normalizedProvider === "openai") {
    return normalizeText(openAiVoice) || DEFAULT_OPENAI_VOICE
  }
  if (normalizedProvider === "elevenlabs") {
    return normalizeText(elevenLabsVoice) || ""
  }
  return normalizeText(tldwVoice) || DEFAULT_TLDW_TTS_VOICE
}

export const useResolvedPersonaVoiceDefaults = (
  personaVoiceDefaults?: PersonaVoiceDefaults | null
): ResolvedPersonaVoiceDefaults => {
  const sttSettings = useSttSettings()
  const {
    voiceChatTriggerPhrases,
    voiceChatAutoResume,
    voiceChatBargeIn
  } = useVoiceChatSettings()
  const [speechToTextLanguage] = useStorage(
    "speechToTextLanguage",
    DEFAULT_STT_LANGUAGE
  )
  const [ttsProvider] = useStorage("ttsProvider", DEFAULT_TTS_PROVIDER)
  const [tldwTtsVoice] = useStorage("tldwTtsVoice", DEFAULT_TLDW_TTS_VOICE)
  const [openAITTSVoice] = useStorage("openAITTSVoice", DEFAULT_OPENAI_VOICE)
  const [elevenLabsVoiceId] = useStorage("elevenLabsVoiceId", "")

  return React.useMemo(() => {
    const resolvedProvider =
      normalizeText(personaVoiceDefaults?.tts_provider) ||
      normalizeText(ttsProvider) ||
      DEFAULT_TTS_PROVIDER
    const personaTriggerPhrases = normalizePhrases(
      personaVoiceDefaults?.voice_chat_trigger_phrases
    )
    const fallbackTriggerPhrases = normalizePhrases(voiceChatTriggerPhrases)

    return {
      sttLanguage:
        normalizeText(personaVoiceDefaults?.stt_language) ||
        normalizeText(String(speechToTextLanguage || "")) ||
        DEFAULT_STT_LANGUAGE,
      sttModel:
        normalizeText(personaVoiceDefaults?.stt_model) ||
        normalizeText(String(sttSettings.model || "")) ||
        "",
      ttsProvider: resolvedProvider,
      ttsVoice:
        normalizeText(personaVoiceDefaults?.tts_voice) ||
        resolveDefaultTtsVoice(
          resolvedProvider,
          String(tldwTtsVoice || ""),
          String(openAITTSVoice || ""),
          String(elevenLabsVoiceId || "")
        ),
      confirmationMode:
        personaVoiceDefaults?.confirmation_mode || DEFAULT_CONFIRMATION_MODE,
      wakeBehavior: normalizeWakeBehavior(personaVoiceDefaults?.wake_behavior),
      voiceChatTriggerPhrases:
        personaTriggerPhrases.length > 0
          ? personaTriggerPhrases
          : fallbackTriggerPhrases,
      autoResume:
        typeof personaVoiceDefaults?.auto_resume === "boolean"
          ? personaVoiceDefaults.auto_resume
          : Boolean(voiceChatAutoResume),
      bargeIn:
        typeof personaVoiceDefaults?.barge_in === "boolean"
          ? personaVoiceDefaults.barge_in
          : Boolean(voiceChatBargeIn),
      autoCommitEnabled:
        typeof personaVoiceDefaults?.auto_commit_enabled === "boolean"
          ? personaVoiceDefaults.auto_commit_enabled
          : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.autoCommitEnabled,
      vadThreshold:
        typeof personaVoiceDefaults?.vad_threshold === "number"
          ? personaVoiceDefaults.vad_threshold
          : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.vadThreshold,
      minSilenceMs:
        typeof personaVoiceDefaults?.min_silence_ms === "number"
          ? personaVoiceDefaults.min_silence_ms
          : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.minSilenceMs,
      turnStopSecs:
        typeof personaVoiceDefaults?.turn_stop_secs === "number"
          ? personaVoiceDefaults.turn_stop_secs
          : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.turnStopSecs,
      minUtteranceSecs:
        typeof personaVoiceDefaults?.min_utterance_secs === "number"
          ? personaVoiceDefaults.min_utterance_secs
          : PERSONA_TURN_DETECTION_BALANCED_DEFAULTS.minUtteranceSecs
    }
  }, [
    elevenLabsVoiceId,
    openAITTSVoice,
    personaVoiceDefaults,
    speechToTextLanguage,
    sttSettings.model,
    tldwTtsVoice,
    ttsProvider,
    voiceChatAutoResume,
    voiceChatBargeIn,
    voiceChatTriggerPhrases
  ])
}
