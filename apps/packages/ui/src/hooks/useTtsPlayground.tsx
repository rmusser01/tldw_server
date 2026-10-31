import { useState, useRef, useEffect, useCallback, type SetStateAction } from "react"
import { splitMessageContent } from "@/utils/tts"
import {
  resolveTtsProviderContext,
  type TtsProviderOverrides,
  type TtsSynthesisResult
} from "@/services/tts-provider"
import { classifyAudioError } from "@/components/Option/Audio/audio-error-classification"
import { useAntdNotification } from "./useAntdNotification"
import { useTranslation } from "react-i18next"

export type TtsPlaygroundSegment = {
  id: string
  index: number
  text: string
  url?: string
  blob?: Blob
  format?: string
  mimeType?: string
  source?: "browser" | "generated"
  requestedBackend?: string
  actualBackend?: string
  fallbackUsed?: boolean
}

export type TtsPlaygroundOverrides = TtsProviderOverrides & {
  splitBy?: string
}

export type TtsPresetKey = "fast" | "balanced" | "quality"

export const TTS_PRESETS: Record<
  TtsPresetKey,
  {
    label: string
    speed: number
    responseFormat: string
    streaming: boolean
    splitBy: string
  }
> = {
  fast: {
    label: "Fast",
    speed: 1.2,
    responseFormat: "mp3",
    streaming: true,
    splitBy: "punctuation"
  },
  balanced: {
    label: "Balanced",
    speed: 1.0,
    responseFormat: "mp3",
    streaming: false,
    splitBy: "punctuation"
  },
  quality: {
    label: "Quality",
    speed: 0.9,
    responseFormat: "wav",
    streaming: false,
    splitBy: "paragraph"
  }
}

const createObjectUrl = (
  audio: TtsSynthesisResult
): { url: string; blob: Blob; format: string; mimeType: string } => {
  const blob = new Blob([audio.buffer], { type: audio.mimeType })
  return {
    url: URL.createObjectURL(blob),
    blob,
    format: audio.format,
    mimeType: audio.mimeType
  }
}

export const useTtsPlayground = () => {
  const [segments, setSegmentsState] = useState<TtsPlaygroundSegment[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationProgress, setGenerationProgress] = useState<{
    completed: number
    total: number
  } | null>(null)
  const notification = useAntdNotification()
  const { t } = useTranslation("playground")

  const segmentsRef = useRef<TtsPlaygroundSegment[]>([])
  const ownedUrls = useRef(new Set<string>())
  const controllerRef = useRef<AbortController | null>(null)
  const mounted = useRef(true)

  const revokeAll = useCallback((urls: Iterable<string>) => {
    for (const url of urls) {
      URL.revokeObjectURL(url)
      ownedUrls.current.delete(url)
    }
  }, [])

  // The public setter also owns URLs produced by streaming and job callers.
  const setSegments = useCallback((next: SetStateAction<TtsPlaygroundSegment[]>) => {
    const value = typeof next === 'function' ? next(segmentsRef.current) : next
    const urls = new Set(value.flatMap(segment => segment.url ? [segment.url] : []))
    if (!mounted.current) {
      revokeAll(urls)
      return
    }
    revokeAll(segmentsRef.current.flatMap(segment =>
      segment.url && !urls.has(segment.url) ? [segment.url] : []))
    urls.forEach(url => ownedUrls.current.add(url))
    segmentsRef.current = value
    setSegmentsState(value)
  }, [revokeAll])

  const clearSegments = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    revokeAll(ownedUrls.current)
    segmentsRef.current = []
    if (mounted.current) {
      setSegmentsState([])
      setIsGenerating(false)
      setGenerationProgress(null)
    }
  }, [revokeAll])

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      clearSegments()
    }
  }, [clearSegments])

  const generateSegments = async (
    text: string,
    overrides?: TtsPlaygroundOverrides
  ): Promise<TtsPlaygroundSegment[]> => {
    clearSegments()
    if (!mounted.current || !text.trim()) return []
    const controller = new AbortController()
    controllerRef.current = controller
    const isCurrent = () => mounted.current && !controller.signal.aborted

    setIsGenerating(true)
    setGenerationProgress(null)
    const createdUrls: string[] = []

    try {
      const context = await resolveTtsProviderContext(text, overrides)
      if (!isCurrent()) return []
      const {
        provider,
        utterance,
        synthesize,
        supported,
        formatInfo
      } = context
      const sentences = splitMessageContent(
        utterance,
        overrides?.splitBy || "punctuation"
      )
      const outSegments: TtsPlaygroundSegment[] = []

      if (!supported) {
        notification.warning({
          message: t(
            "tts.unsupportedProviderTitle",
            "Unsupported TTS provider"
          ),
          description: t(
            "tts.unsupportedProviderDescription",
            'The provider "{{provider}}" is not yet supported in the playground player.',
            { provider }
          )
        })
        setSegments([])
        return []
      }

      if (provider === "browser") {
        notification.info({
          message: t(
            "tts.browserInfoTitle",
            "Browser TTS uses system audio"
          ),
          description: t(
            "tts.browserInfoDescription",
            "Browser TTS plays using your system synthesizer and does not expose a downloadable audio file. Use the segment controls below, or switch providers for a track list and player."
          )
        })
        const browserSegments = sentences.map((sentence, i) => ({
          id: `browser-${i}`,
          index: i,
          text: sentence,
          source: "browser" as const
        }))
        setSegments(browserSegments)
        return browserSegments
      }

      if (!synthesize) {
        notification.warning({
          message: t(
            "tts.unsupportedProviderTitle",
            "Unsupported TTS provider"
          ),
          description: t(
            "tts.unsupportedProviderDescription",
            'The provider "{{provider}}" is not yet supported in the playground player.',
            { provider }
          )
        })
        setSegments([])
        return []
      }

      if (provider === "tldw" && formatInfo?.isFallback) {
        notification.warning({
          message: t(
            "tts.unsupportedFormatTitle",
            "Unsupported audio format"
          ),
          description: t(
            "tts.unsupportedFormatDescription",
            'The response format "{{format}}" is not supported. Falling back to MP3.',
            { format: formatInfo.requested }
          )
        })
      }

      const idPrefix = provider === "elevenlabs" ? "eleven" : provider
      setGenerationProgress({ completed: 0, total: sentences.length })
      for (let i = 0; i < sentences.length; i++) {
        const audio = await synthesize(sentences[i], { signal: controller.signal })
        if (!isCurrent()) return []
        const created = createObjectUrl(audio)
        createdUrls.push(created.url)
        ownedUrls.current.add(created.url)
        outSegments.push({
          id: `${idPrefix}-${i}`,
          index: i,
          text: sentences[i],
          url: created.url,
          blob: created.blob,
          format: created.format,
          mimeType: created.mimeType,
          source: "generated",
          requestedBackend: overrides?.tldwBackend,
          actualBackend: audio.actualBackend,
          fallbackUsed: audio.fallbackUsed
        })
        setGenerationProgress({ completed: i + 1, total: sentences.length })
      }

      // We do not apply playbackSpeed here because <audio> controls can be used directly.
      setSegments(outSegments)
      return outSegments
    } catch (error) {
      revokeAll(createdUrls.filter(url => ownedUrls.current.has(url)))
      if (!isCurrent()) return []
      setSegments([])
      const classified = classifyAudioError(error)
      notification.error({
        message:
          classified.title || t("tts.generateErrorTitle", "Error generating audio"),
        description:
          classified.recovery ||
          t(
            "tts.generateErrorDescription",
            "Something went wrong while generating TTS audio."
          )
      })
      return []
    } finally {
      if (isCurrent()) {
        controllerRef.current = null
        setIsGenerating(false)
        setGenerationProgress(null)
      }
    }
  }

  return {
    segments,
    isGenerating,
    generationProgress,
    generateSegments,
    clearSegments,
    setSegments
  }
}
