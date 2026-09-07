import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const storageState = vi.hoisted(() => ({
  values: {
    speechToTextLanguage: "en-US",
    ttsProvider: "tldw",
    tldwTtsVoice: "Bella",
    openAITTSVoice: "alloy",
    elevenLabsVoiceId: "voice-eleven"
  } as Record<string, unknown>
}))

const voiceChatState = vi.hoisted(() => ({
  voiceChatTriggerPhrases: ["hey assistant"],
  voiceChatAutoResume: true,
  voiceChatBargeIn: false
}))

const sttState = vi.hoisted(() => ({
  model: "parakeet"
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    key in storageState.values ? storageState.values[key] : defaultValue
  ]
}))

vi.mock("@/hooks/useVoiceChatSettings", () => ({
  useVoiceChatSettings: () => voiceChatState
}))

vi.mock("@/hooks/useSttSettings", () => ({
  useSttSettings: () => ({
    model: sttState.model,
    temperature: 0,
    task: "transcribe",
    responseFormat: "json",
    timestampGranularities: "segment",
    prompt: "",
    useSegmentation: false,
    segK: 3,
    segMinSegmentSize: 20,
    segLambdaBalance: 0.5,
    segUtteranceExpansionWidth: 2,
    segEmbeddingsProvider: "",
    segEmbeddingsModel: ""
  })
}))

import { useResolvedPersonaVoiceDefaults } from "../useResolvedPersonaVoiceDefaults"

describe("useResolvedPersonaVoiceDefaults", () => {
  beforeEach(() => {
    storageState.values = {
      speechToTextLanguage: "en-US",
      ttsProvider: "tldw",
      tldwTtsVoice: "Bella",
      openAITTSVoice: "alloy",
      elevenLabsVoiceId: "voice-eleven"
    }
    voiceChatState.voiceChatTriggerPhrases = ["hey assistant"]
    voiceChatState.voiceChatAutoResume = true
    voiceChatState.voiceChatBargeIn = false
    sttState.model = "parakeet"
  })

  it("keeps explicit Persona provider voice unset while inheriting non-voice browser settings", () => {
    const { result } = renderHook(() =>
      useResolvedPersonaVoiceDefaults({
        stt_language: "fr-FR",
        tts_provider: "openai",
        confirmation_mode: "always",
        voice_chat_trigger_phrases: ["bonjour helper"],
        wake_behavior: "continuous",
        auto_resume: false,
        auto_commit_enabled: false,
        vad_threshold: 0.61,
        min_silence_ms: 640,
        turn_stop_secs: 0.48,
        min_utterance_secs: 0.82
      })
    )

    expect(result.current).toEqual({
      sttLanguage: "fr-FR",
      sttModel: "parakeet",
      ttsProvider: "openai",
      ttsVoice: "",
      confirmationMode: "always",
      voiceChatTriggerPhrases: ["bonjour helper"],
      wakeBehavior: "continuous",
      autoResume: false,
      bargeIn: false,
      autoCommitEnabled: false,
      vadThreshold: 0.61,
      minSilenceMs: 640,
      turnStopSecs: 0.48,
      minUtteranceSecs: 0.82
    })
  })

  it("inherits the browser provider without overriding its server voice default", () => {
    storageState.values.ttsProvider = "elevenlabs"
    voiceChatState.voiceChatTriggerPhrases = ["okay helper", "status check"]
    voiceChatState.voiceChatAutoResume = false
    voiceChatState.voiceChatBargeIn = true
    sttState.model = "whisper-1"

    const { result } = renderHook(() => useResolvedPersonaVoiceDefaults(null))

    expect(result.current).toEqual({
      sttLanguage: "en-US",
      sttModel: "whisper-1",
      ttsProvider: "elevenlabs",
      ttsVoice: "",
      confirmationMode: "destructive_only",
      voiceChatTriggerPhrases: ["okay helper", "status check"],
      wakeBehavior: "one_shot",
      autoResume: false,
      bargeIn: true,
      autoCommitEnabled: true,
      vadThreshold: 0.5,
      minSilenceMs: 250,
      turnStopSecs: 0.2,
      minUtteranceSecs: 0.4
    })
  })

  it("leaves the fresh-profile voice to the selected provider", () => {
    storageState.values = {
      speechToTextLanguage: "en-US"
    }

    const { result } = renderHook(() => useResolvedPersonaVoiceDefaults(null))

    expect(result.current.ttsProvider).toBe("tldw")
    expect(result.current.ttsVoice).toBe("")
    expect(result.current.wakeBehavior).toBe("one_shot")
  })

  it.each(["browser", "kokoro", "piper", "custom-server"])(
    "does not borrow the tldw voice for %s",
    (provider) => {
      const { result } = renderHook(() =>
        useResolvedPersonaVoiceDefaults({ tts_provider: provider })
      )
      expect(result.current.ttsVoice).toBe("")
      expect(result.current.ttsProvider).toBe(provider)
    }
  )

  it("resolves an optional persona TTS model without borrowing another provider model", () => {
    const { result } = renderHook(() =>
      useResolvedPersonaVoiceDefaults({
        tts_provider: "piper",
        tts_model: " en_US-lessac-medium "
      })
    )
    expect(result.current.ttsModel).toBe("en_US-lessac-medium")
  })

  it("resolves explicit persona wake behavior", () => {
    const { result } = renderHook(() =>
      useResolvedPersonaVoiceDefaults({
        wake_behavior: "push_to_talk_after_wake"
      })
    )

    expect(result.current.wakeBehavior).toBe("push_to_talk_after_wake")
  })
})


it.each(["openai", "elevenlabs", "tldw"])("keeps a blank %s voice unset instead of injecting browser preferences", provider => {
  const {result}=renderHook(()=>useResolvedPersonaVoiceDefaults({tts_provider:provider,tts_voice:"   "}))
  expect(result.current.ttsVoice).toBe("")
})

it("preserves an explicit Persona voice",()=>{
 const {result}=renderHook(()=>useResolvedPersonaVoiceDefaults({tts_provider:"openai",tts_voice:" nova "}))
 expect(result.current.ttsVoice).toBe("nova")
})
