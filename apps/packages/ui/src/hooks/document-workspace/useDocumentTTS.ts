import { useState, useCallback, useRef, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchTldwVoices, type TldwVoice } from "@/services/tldw/audio-voices"
import {
  DEFAULT_TLDW_TTS_VOICE,
  getTldwTTSModel,
  getTldwTTSVoice
} from "@/services/tts"

/**
 * TTS Voice information
 */
export interface TTSVoice {
  id: string
  name: string
  provider: string
  language?: string
  gender?: string
}

/**
 * Convert TldwVoice to TTSVoice
 */
function toTTSVoice(voice: TldwVoice): TTSVoice {
  return {
    id: voice.voice_id || voice.id || voice.name || "",
    name: voice.name || voice.id || voice.voice_id || "",
    provider: voice.provider || "unknown"
  }
}

/**
 * TTS playback state
 */
export interface TTSState {
  isPlaying: boolean
  isPaused: boolean
  isLoading: boolean
  error: string | null
  currentText: string | null
  /** Last text that was spoken, persists after playback ends for replay */
  lastSpokenText: string | null
}

/**
 * Hook return type
 */
export interface UseDocumentTTSReturn {
  // State
  state: TTSState
  voice: string
  speed: number
  volume: number
  progress: number
  audioUrl: string | null
  voices: TTSVoice[]
  voicesLoading: boolean

  // Actions
  speak: (text: string) => Promise<void>
  pause: () => void
  resume: () => void
  stop: () => void
  setVoice: (voiceId: string) => void
  setSpeed: (speed: number) => void
  setVolume: (volume: number) => void
}

const DEFAULT_VOICE = DEFAULT_TLDW_TTS_VOICE
const DEFAULT_SPEED = 1.0

const readStoredVoicePreference = (): string => {
  if (typeof window === "undefined") return ""
  try {
    return localStorage.getItem("tts-voice") || ""
  } catch {
    return ""
  }
}

/**
 * Hook for text-to-speech playback using the TTS API.
 *
 * Features:
 * - Stream audio from /api/v1/audio/speech
 * - Voice selection from catalog
 * - Speed control
 * - Play/pause/stop controls
 * - Multiple provider support
 */
export function useDocumentTTS(): UseDocumentTTSReturn {
  const [voice, setVoiceState] = useState(() => readStoredVoicePreference())

  const [speed, setSpeedState] = useState(() => {
    if (typeof window === "undefined") return DEFAULT_SPEED
    try {
      const saved = localStorage.getItem("tts-speed")
      return saved ? parseFloat(saved) : DEFAULT_SPEED
    } catch { return DEFAULT_SPEED }
  })

  const [volume, setVolumeState] = useState(() => {
    if (typeof window === "undefined") return 1
    try {
      const saved = localStorage.getItem("tts-volume")
      return saved ? parseFloat(saved) : 1
    } catch { return 1 }
  })

  const [progress, setProgress] = useState(0)

  const [state, setState] = useState<TTSState>({
    isPlaying: false,
    isPaused: false,
    isLoading: false,
    error: null,
    currentText: null,
    lastSpokenText: null
  })

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioUrlRef = useRef<string | null>(null)

  // Fetch available voices
  const { data: voicesData, isLoading: voicesLoading } = useQuery({
    queryKey: ["tts-voices"],
    queryFn: async (): Promise<TTSVoice[]> => {
      try {
        const tldwVoices = await fetchTldwVoices()
        if (tldwVoices.length > 0) {
          return tldwVoices.map(toTTSVoice)
        }
        // Return provider-compatible fallback voices if the catalog is empty.
        return [
          { id: "Bella", name: "Bella", provider: "kitten_tts" },
          { id: "Jasper", name: "Jasper", provider: "kitten_tts" },
          { id: "Luna", name: "Luna", provider: "kitten_tts" },
          { id: "Leo", name: "Leo", provider: "kitten_tts" }
        ]
      } catch (e) {
        console.error("Failed to fetch TTS voices:", e)
        // Return provider-compatible fallback voices if the catalog request fails.
        return [
          { id: "Bella", name: "Bella", provider: "kitten_tts" },
          { id: "Jasper", name: "Jasper", provider: "kitten_tts" },
          { id: "Luna", name: "Luna", provider: "kitten_tts" },
          { id: "Leo", name: "Leo", provider: "kitten_tts" }
        ]
      }
    },
    staleTime: 30 * 60 * 1000, // 30 minutes
    retry: 1
  })

  const voices = voicesData || []

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
        audioUrlRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const loadConfiguredVoice = async () => {
      if (typeof window === "undefined") return

      try {
        const storedVoiceBefore = readStoredVoicePreference()
        if (storedVoiceBefore) {
          setVoiceState(storedVoiceBefore)
          return
        }

        const configuredVoice = await getTldwTTSVoice()
        if (cancelled) return

        const storedVoiceAfter = readStoredVoicePreference()
        if (storedVoiceAfter) {
          setVoiceState(storedVoiceAfter)
          return
        }

        if (configuredVoice) {
          setVoiceState(configuredVoice)
        }
      } catch {
        // Ignore storage/config lookup errors and keep the local fallback.
      }
    }

    void loadConfiguredVoice()

    return () => {
      cancelled = true
    }
  }, [])

  // Speak text
  const speak = useCallback(async (text: string) => {
    // Stop any current playback
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }

    setProgress(0)

    setState((prev) => ({
      isPlaying: false,
      isPaused: false,
      isLoading: true,
      error: null,
      currentText: text,
      lastSpokenText: text
    }))

    try {
      const model = await getTldwTTSModel()
      const fallbackVoice = await getTldwTTSVoice()
      const resolvedVoice =
        readStoredVoicePreference() || voice || fallbackVoice || DEFAULT_VOICE

      // Call TTS API
      const response = await fetch("/api/v1/audio/speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          input: text,
          voice: resolvedVoice,
          model,
          speed: speed,
          response_format: "mp3"
        })
      })

      if (!response.ok) {
        throw new Error(`TTS request failed: ${response.statusText}`)
      }

      // Get audio blob
      const audioBlob = await response.blob()
      const audioUrl = URL.createObjectURL(audioBlob)
      audioUrlRef.current = audioUrl

      // Create and play audio
      const audio = new Audio(audioUrl)
      audioRef.current = audio

      audio.onended = () => {
        setState((prev) => ({
          ...prev,
          isPlaying: false,
          isPaused: false,
          currentText: null,
          lastSpokenText: prev.lastSpokenText
        }))
      }

      audio.onerror = () => {
        setState((prev) => ({
          ...prev,
          isPlaying: false,
          isLoading: false,
          error: "Failed to play audio"
        }))
      }

      audio.onplay = () => {
        setState((prev) => ({
          ...prev,
          isPlaying: true,
          isLoading: false,
          error: null
        }))
      }

      audio.onpause = () => {
        if (!audio.ended) {
          setState((prev) => ({
            ...prev,
            isPlaying: false,
            isPaused: true
          }))
        }
      }

      audio.ontimeupdate = () => {
        if (audio.duration > 0) {
          setProgress((audio.currentTime / audio.duration) * 100)
        }
      }

      audio.volume = volume

      await audio.play()
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "TTS failed"
      setState((prev) => ({
        isPlaying: false,
        isPaused: false,
        isLoading: false,
        error: errorMessage,
        currentText: null,
        lastSpokenText: prev.lastSpokenText
      }))
    }
  }, [voice, speed, volume])

  // Pause playback
  const pause = useCallback(() => {
    if (audioRef.current && !audioRef.current.paused) {
      audioRef.current.pause()
    }
  }, [])

  // Resume playback
  const resume = useCallback(() => {
    if (audioRef.current && audioRef.current.paused) {
      audioRef.current.play()
    }
  }, [])

  // Stop playback
  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current = null
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
    setProgress(0)
    setState((prev) => ({
      isPlaying: false,
      isPaused: false,
      isLoading: false,
      error: null,
      currentText: null,
      lastSpokenText: prev.lastSpokenText
    }))
  }, [])

  // Set voice with persistence
  const handleSetVoice = useCallback((voiceId: string) => {
    setVoiceState(voiceId)
    try {
      localStorage.setItem("tts-voice", voiceId)
    } catch (e) {
      // Ignore storage errors
    }
  }, [])

  // Set volume with persistence
  const setVolume = useCallback((newVolume: number) => {
    const clamped = Math.max(0, Math.min(1, newVolume))
    setVolumeState(clamped)
    if (audioRef.current) {
      audioRef.current.volume = clamped
    }
    try {
      localStorage.setItem("tts-volume", String(clamped))
    } catch { /* ignore */ }
  }, [])

  // Set speed with persistence
  const setSpeed = useCallback((newSpeed: number) => {
    const clampedSpeed = Math.max(0.25, Math.min(4, newSpeed))
    setSpeedState(clampedSpeed)
    try {
      localStorage.setItem("tts-speed", String(clampedSpeed))
    } catch (e) {
      // Ignore storage errors
    }
  }, [])

  return {
    state,
    voice: voice || DEFAULT_VOICE,
    speed,
    volume,
    progress,
    audioUrl: audioUrlRef.current,
    voices,
    voicesLoading,
    speak,
    pause,
    resume,
    stop,
    setVoice: handleSetVoice,
    setSpeed,
    setVolume
  }
}

export default useDocumentTTS
