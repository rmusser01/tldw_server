import React from "react"
import { useSetting } from "@/hooks/useSetting"
import {
  VOICE_CHAT_AUTO_RESUME_SETTING,
  VOICE_CHAT_BARGE_IN_SETTING,
  VOICE_CHAT_MODEL_SETTING,
  VOICE_CHAT_PAUSE_MS_SETTING,
  VOICE_CHAT_TRIGGER_PHRASES_SETTING,
  VOICE_CHAT_TTS_MODE_SETTING,
  type VoiceChatTtsMode
} from "@/services/settings/ui-settings"

// The "voice chat on" flag is deliberately NOT persisted: a stored `true`
// reopened the microphone + WebSocket on every page load with no user gesture.
// Module-level state shared by all consumers in the page; a reload resets it.
let voiceChatEnabledState = false
const voiceChatEnabledListeners = new Set<() => void>()
const subscribeVoiceChatEnabled = (listener: () => void) => {
  voiceChatEnabledListeners.add(listener)
  return () => {
    voiceChatEnabledListeners.delete(listener)
  }
}
const getVoiceChatEnabled = () => voiceChatEnabledState
const setVoiceChatEnabledState = (value: boolean | ((prev: boolean) => boolean)) => {
  const next = typeof value === "function" ? value(voiceChatEnabledState) : value
  if (next === voiceChatEnabledState) return
  voiceChatEnabledState = next
  voiceChatEnabledListeners.forEach((listener) => listener())
}

export const useVoiceChatSettings = () => {
  const voiceChatEnabled = React.useSyncExternalStore(
    subscribeVoiceChatEnabled,
    getVoiceChatEnabled,
    () => false
  )
  const setVoiceChatEnabled = setVoiceChatEnabledState
  const [voiceChatModel, setVoiceChatModel] = useSetting(
    VOICE_CHAT_MODEL_SETTING
  )
  const [voiceChatPauseMs, setVoiceChatPauseMs] = useSetting(
    VOICE_CHAT_PAUSE_MS_SETTING
  )
  const [voiceChatTriggerPhrases, setVoiceChatTriggerPhrases] = useSetting(
    VOICE_CHAT_TRIGGER_PHRASES_SETTING
  )
  const [voiceChatAutoResume, setVoiceChatAutoResume] = useSetting(
    VOICE_CHAT_AUTO_RESUME_SETTING
  )
  const [voiceChatBargeIn, setVoiceChatBargeIn] = useSetting(
    VOICE_CHAT_BARGE_IN_SETTING
  )
  const [voiceChatTtsMode, setVoiceChatTtsMode] = useSetting(
    VOICE_CHAT_TTS_MODE_SETTING
  )

  const normalizedTriggerPhrases = React.useMemo(
    () =>
      (voiceChatTriggerPhrases || [])
        .map((phrase) => phrase.trim())
        .filter(Boolean),
    [voiceChatTriggerPhrases]
  )

  const wrappedSetVoiceChatTriggerPhrases = React.useCallback(
    (value: string[] | ((prev: string[]) => string[])) => {
      if (typeof value === "function") {
        setVoiceChatTriggerPhrases((prev) => value(prev || []))
        return
      }
      setVoiceChatTriggerPhrases(value)
    },
    [setVoiceChatTriggerPhrases]
  )

  return React.useMemo(
    () => ({
      voiceChatEnabled,
      setVoiceChatEnabled,
      voiceChatModel,
      setVoiceChatModel,
      voiceChatPauseMs,
      setVoiceChatPauseMs,
      voiceChatTriggerPhrases: normalizedTriggerPhrases,
      setVoiceChatTriggerPhrases: wrappedSetVoiceChatTriggerPhrases,
      voiceChatAutoResume,
      setVoiceChatAutoResume,
      voiceChatBargeIn,
      setVoiceChatBargeIn,
      voiceChatTtsMode: voiceChatTtsMode as VoiceChatTtsMode,
      setVoiceChatTtsMode
    }),
    [
      voiceChatEnabled,
      setVoiceChatEnabled,
      voiceChatModel,
      setVoiceChatModel,
      voiceChatPauseMs,
      setVoiceChatPauseMs,
      normalizedTriggerPhrases,
      wrappedSetVoiceChatTriggerPhrases,
      voiceChatAutoResume,
      setVoiceChatAutoResume,
      voiceChatBargeIn,
      setVoiceChatBargeIn,
      voiceChatTtsMode,
      setVoiceChatTtsMode
    ]
  )
}
