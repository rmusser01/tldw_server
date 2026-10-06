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

/** Detach all listeners we attach in speak() so a stopped/replaced audio element
 * can't fire onpause/onerror/ontimeupdate and clobber state after teardown. */
const detachAudioListeners = (audio: HTMLAudioElement | null): void => {
  if (!audio) return
  audio.onended = null
  audio.onerror = null
  audio.onplay = null
  audio.onpause = null
  audio.ontimeupdate = null
}

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

  const controllerRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)
  const releasePlayback = useCallback(() => {
    if (audioRef.current) {
      detachAudioListeners(audioRef.current)
      audioRef.current.pause()
      audioRef.current.src = ''
      audioRef.current.load?.()
      audioRef.current = null
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
  }, [])

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

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      controllerRef.current?.abort()
      releasePlayback()
    }
  }, [releasePlayback])

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
    controllerRef.current?.abort()
    releasePlayback()
    if (!mountedRef.current) return
    const controller = new AbortController()
    controllerRef.current = controller

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
      if (controller.signal.aborted) return
      const resolvedVoice =
        readStoredVoicePreference() || voice || fallbackVoice || DEFAULT_VOICE

      // Call TTS API
      const response = await fetch("/api/v1/audio/speech", {
        method: "POST",
        signal: controller.signal,
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

      if (controller.signal.aborted) return
      if (!response.ok) {
        throw new Error(`TTS request failed: ${response.statusText}`)
      }

      // Get audio blob
      const audioBlob = await response.blob()
      if (controller.signal.aborted) return
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
        // Free the blob URL on the error path too (onended isn't fired on error).
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current)
          audioUrlRef.current = null
        }
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

      try {
        await audio.play()
      } catch (playErr) {
        // A Stop/pause issued right after Speak aborts the play() promise; that
        // is expected teardown, not a playback error worth surfacing.
        if (playErr instanceof DOMException && playErr.name === "AbortError") {
          return
        }
        throw playErr
      }
    } catch (err) {
      if (controller.signal.aborted) return
      releasePlayback()
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
  }, [voice, speed, volume, releasePlayback])

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
    controllerRef.current?.abort()
    releasePlayback()
    setProgress(0)
    setState((prev) => ({
      isPlaying: false,
      isPaused: false,
      isLoading: false,
      error: null,
      currentText: null,
      lastSpokenText: prev.lastSpokenText
    }))
  }, [releasePlayback])

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
