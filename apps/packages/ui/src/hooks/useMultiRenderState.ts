import { useCallback, useEffect, useRef, useState } from "react"
import {
  resolveTtsProviderContext,
  type TtsProviderOverrides
} from "@/services/tts-provider"
import type { RenderStripConfig, RenderStripState } from "@/components/Option/Speech/RenderStrip"
import {
  classifyAudioError,
  type AudioErrorClassification
} from "@/components/Option/Audio/audio-error-classification"
import {
  buildTtsResultMetadata,
  type TtsResultMetadata
} from "@/components/Option/Audio/comparison-provenance"

export type RenderEntry = {
  id: string
  config: RenderStripConfig
  state: RenderStripState
  audioUrl?: string
  audioBlob?: Blob
  errorMessage?: string
  errorSettingsHref?: "/settings/speech"
  progress?: number
  metadata?: TtsResultMetadata
  disabled?: boolean
}

let nextId = 1
const genId = () => `render-${Date.now()}-${nextId++}`

const classifyRenderError = (error: unknown): AudioErrorClassification => {
  return classifyAudioError(error)
}

const formatRenderErrorMessage = (classified: AudioErrorClassification): string =>
  `${classified.title}. ${classified.recovery}`

const configToOverrides = (config: RenderStripConfig): TtsProviderOverrides => {
  const overrides: TtsProviderOverrides = { provider: config.provider }
  if (config.provider === "tldw") {
    overrides.tldwModel = config.model
    overrides.tldwVoice = config.voice
    overrides.tldwResponseFormat = config.format
    overrides.tldwSpeed = config.speed
  } else if (config.provider === "openai") {
    overrides.openAiModel = config.model
    overrides.openAiVoice = config.voice
    overrides.openAiSpeed = config.speed
  } else if (config.provider === "elevenlabs") {
    overrides.elevenLabsModel = config.model
    overrides.elevenLabsVoiceId = config.voice
    overrides.elevenLabsSpeed = config.speed
  }
  return overrides
}

const cloneConfig = (config: RenderStripConfig): RenderStripConfig => ({
  ...config
})

export const useMultiRenderState = () => {
  const [renders, setRenders] = useState<RenderEntry[]>([])
  const [playingId, setPlayingId] = useState<string | null>(null)
  const objectUrlsRef = useRef<Map<string, string>>(new Map())
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map())

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      for (const url of objectUrlsRef.current.values()) {
        try { URL.revokeObjectURL(url) } catch {}
      }
      objectUrlsRef.current.clear()
      for (const ctrl of abortControllersRef.current.values()) {
        try { ctrl.abort() } catch {}
      }
      abortControllersRef.current.clear()
    }
  }, [])

  const addRender = useCallback((config: RenderStripConfig): string => {
    const id = genId()
    setRenders((prev) => [
      ...prev,
      { id, config, state: "idle" }
    ])
    return id
  }, [])

  const removeRender = useCallback((id: string) => {
    // Revoke object URL
    const url = objectUrlsRef.current.get(id)
    if (url) {
      try { URL.revokeObjectURL(url) } catch {}
      objectUrlsRef.current.delete(id)
    }
    // Abort any in-progress generation
    const ctrl = abortControllersRef.current.get(id)
    if (ctrl) {
      try { ctrl.abort() } catch {}
      abortControllersRef.current.delete(id)
    }
    setRenders((prev) => prev.filter((r) => r.id !== id))
    setPlayingId((prev) => (prev === id ? null : prev))
  }, [])

  const updateRender = useCallback(
    (id: string, updates: Partial<RenderEntry>) => {
      setRenders((prev) =>
        prev.map((r) => (r.id === id ? { ...r, ...updates } : r))
      )
    },
    []
  )

  const updateConfig = useCallback(
    (id: string, config: RenderStripConfig) => {
      setRenders((prev) =>
        prev.map((r) => (r.id === id ? { ...r, config } : r))
      )
    },
    []
  )

  const generateRender = useCallback(
    async (id: string, text: string) => {
      if (!text.trim()) return

      const entry = renders.find((r) => r.id === id)
      if (!entry) return

      const createdAt = new Date().toISOString()
      updateRender(id, {
        state: "generating",
        progress: 0,
        errorMessage: undefined,
        metadata: buildTtsResultMetadata(text, createdAt)
      })

      const controller = new AbortController()
      abortControllersRef.current.set(id, controller)
      const start = performance.now()

      try {
        const overrides = configToOverrides(entry.config)
        const context = await resolveTtsProviderContext(text, overrides)

        if (!context.supported || !context.synthesize) {
          const classified = classifyRenderError(
            new Error(`Provider "${entry.config.provider}" is not supported`)
          )
          updateRender(id, {
            state: "error",
            errorMessage: formatRenderErrorMessage(classified),
            errorSettingsHref: classified.settingsHref,
            metadata: buildTtsResultMetadata(text, createdAt, {
              clientLatencyMs: performance.now() - start
            })
          })
          return
        }

        if (controller.signal.aborted) return

        const audio = await context.synthesize(context.utterance)
        const clientLatencyMs = performance.now() - start

        if (controller.signal.aborted) return

        const blob = new Blob([audio.buffer], { type: audio.mimeType })
        const url = URL.createObjectURL(blob)

        // Revoke old URL if any
        const oldUrl = objectUrlsRef.current.get(id)
        if (oldUrl) {
          try { URL.revokeObjectURL(oldUrl) } catch {}
        }
        objectUrlsRef.current.set(id, url)

        updateRender(id, {
          state: "ready",
          audioUrl: url,
          audioBlob: blob,
          progress: 100,
          metadata: buildTtsResultMetadata(text, createdAt, {
            audioSizeBytes: blob.size,
            clientLatencyMs
          })
        })
      } catch (error) {
        if (controller.signal.aborted) return
        const classified = classifyRenderError(error)
        updateRender(id, {
          state: "error",
          errorMessage: formatRenderErrorMessage(classified),
          errorSettingsHref: classified.settingsHref,
          metadata: buildTtsResultMetadata(text, createdAt, {
            clientLatencyMs: performance.now() - start
          })
        })
      } finally {
        abortControllersRef.current.delete(id)
      }
    },
    [renders, updateRender]
  )

  const generateAll = useCallback(
    async (text: string) => {
      const pending = renders.filter(
        (r) => !r.disabled && (r.state === "idle" || r.state === "error")
      )
      await Promise.allSettled(
        pending.map((r) => generateRender(r.id, text))
      )
    },
    [renders, generateRender]
  )

  const duplicateRender = useCallback((id: string) => {
    setRenders((prev) => {
      const source = prev.find((render) => render.id === id)
      if (!source) return prev
      return [
        ...prev,
        {
          id: genId(),
          config: cloneConfig(source.config),
          state: "idle" as const,
          disabled: false
        }
      ]
    })
  }, [])

  const setRenderDisabled = useCallback((id: string, disabled: boolean) => {
    setRenders((prev) =>
      prev.map((render) =>
        render.id === id ? { ...render, disabled } : render
      )
    )
  }, [])

  const clearAll = useCallback(() => {
    for (const url of objectUrlsRef.current.values()) {
      try { URL.revokeObjectURL(url) } catch {}
    }
    objectUrlsRef.current.clear()
    for (const ctrl of abortControllersRef.current.values()) {
      try { ctrl.abort() } catch {}
    }
    abortControllersRef.current.clear()
    setRenders([])
    setPlayingId(null)
  }, [])

  // Play-one-at-a-time: setting playingId pauses all others
  const startPlaying = useCallback((id: string) => {
    setPlayingId(id)
  }, [])

  const stopPlaying = useCallback((id: string) => {
    setPlayingId((prev) => (prev === id ? null : prev))
  }, [])

  // Sequential play queue: stores IDs to play in order.
  // When the current strip ends (via onEnd callback from UnifiedAudioPlayer),
  // the next strip in the queue is activated by setting playingId.
  const playQueueRef = useRef<string[]>([])

  const advancePlayQueue = useCallback(() => {
    const next = playQueueRef.current.shift()
    if (next) {
      // 1-second pause between strips
      setTimeout(() => setPlayingId(next), 1000)
    } else {
      setPlayingId(null)
    }
  }, [])

  const playAllSequentially = useCallback(() => {
    const readyStrips = renders.filter(
      (r) => r.state === "ready" && r.audioUrl
    )
    if (readyStrips.length === 0) return

    // Queue all but the first; start the first immediately
    playQueueRef.current = readyStrips.slice(1).map((r) => r.id)
    setPlayingId(readyStrips[0].id)
  }, [renders])

  // Called by RenderStrip when its UnifiedAudioPlayer emits onEnd
  const handleStripEnded = useCallback(
    (id: string) => {
      // If this strip was playing as part of a sequential queue, advance
      if (playQueueRef.current.length > 0) {
        advancePlayQueue()
      } else {
        // Single strip ended naturally — clear playingId
        setPlayingId((prev) => (prev === id ? null : prev))
      }
    },
    [advancePlayQueue]
  )

  const hasIdle = renders.some(
    (r) => !r.disabled && (r.state === "idle" || r.state === "error")
  )
  const hasReady = renders.some((r) => r.state === "ready")
  const isAnyGenerating = renders.some((r) => r.state === "generating")

  return {
    renders,
    playingId,
    addRender,
    removeRender,
    updateRender,
    updateConfig,
    generateRender,
    generateAll,
    duplicateRender,
    setRenderDisabled,
    clearAll,
    startPlaying,
    stopPlaying,
    playAllSequentially,
    handleStripEnded,
    hasIdle,
    hasReady,
    isAnyGenerating
  }
}
