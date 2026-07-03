import { useCallback, useRef } from "react"
import { useTranslation } from "react-i18next"
import { resolveTtsProviderContext } from "@/services/tts-provider"
import { applyVoiceSpeedOverrides, type TtsVoiceConfig } from "@/utils/tts-speed"
import { useAntdNotification } from "./useAntdNotification"
import {
  useAudiobookStudioStore,
  type AudioChapter
} from "@/store/audiobook-studio"

type GenerateChapterOptions = {
  chapter: AudioChapter
  overrides?: TtsVoiceConfig
  onProgress?: (chapterId: string, progress: number) => void
}

type GenerateAllOptions = {
  chapters: AudioChapter[]
  overrides?: TtsVoiceConfig
  onChapterStart?: (chapterId: string) => void
  onChapterComplete?: (chapterId: string) => void
  onChapterError?: (chapterId: string, error: Error) => void
}

export const useAudiobookGeneration = () => {
  const { t } = useTranslation(["playground", "audiobook"])
  const notification = useAntdNotification()
  const abortControllerRef = useRef<AbortController | null>(null)

  const updateChapter = useAudiobookStudioStore((s) => s.updateChapter)
  const setIsGenerating = useAudiobookStudioStore((s) => s.setIsGenerating)
  const setCurrentGeneratingId = useAudiobookStudioStore(
    (s) => s.setCurrentGeneratingId
  )
  const setGenerationQueue = useAudiobookStudioStore(
    (s) => s.setGenerationQueue
  )
  const getGenerationQueue = () => useAudiobookStudioStore.getState().generationQueue
  const resolveAudioExtension = (type: string) => {
    const normalized = type.toLowerCase()
    if (normalized.includes("wav")) return "wav"
    if (normalized.includes("opus")) return "opus"
    if (normalized.includes("aac")) return "aac"
    if (normalized.includes("flac")) return "flac"
    if (normalized.includes("pcm") || normalized.includes("l16")) return "pcm"
    if (normalized.includes("ogg")) return "ogg"
    return "mp3"
  }

  const generateChapterAudio = useCallback(
    async ({
      chapter,
      overrides
    }: GenerateChapterOptions): Promise<Blob | null> => {
      try {
        const effectiveOverrides = applyVoiceSpeedOverrides({
          ...chapter.voiceConfig,
          ...overrides
        })

        const context = await resolveTtsProviderContext(
          chapter.content,
          effectiveOverrides
        )

        if (!context.supported) {
          throw new Error(
            `TTS provider "${context.provider}" is not supported`
          )
        }

        if (context.provider === "browser") {
          notification.warning({
            message: t(
              "audiobook:browserTtsWarning",
              "Browser TTS cannot export audio"
            ),
            description: t(
              "audiobook:browserTtsWarningDesc",
              "Browser TTS streams audio directly and cannot be saved. Please switch to a different TTS provider for audiobook generation."
            )
          })
          return null
        }

        if (!context.synthesize) {
          throw new Error("TTS synthesis function not available")
        }

        // Thread the abort signal through so cancelGeneration() aborts the
        // in-flight chapter synthesis, not just the between-chapter poll.
        const result = await context.synthesize(context.utterance, {
          signal: abortControllerRef.current?.signal
        })
        const blob = new Blob([result.buffer], { type: result.mimeType })
        return blob
      } catch (error) {
        console.error("Failed to generate chapter audio:", error)
        throw error
      }
    },
    [notification, t]
  )

  const generateSingleChapter = useCallback(
    async (
      chapter: AudioChapter,
      overrides?: TtsVoiceConfig
    ): Promise<boolean> => {
      setCurrentGeneratingId(chapter.id)
      updateChapter(chapter.id, { status: "generating", errorMessage: undefined })

      try {
        const blob = await generateChapterAudio({ chapter, overrides })

        if (!blob) {
          updateChapter(chapter.id, { status: "pending", errorMessage: undefined })
          return false
        }

        const url = URL.createObjectURL(blob)
        if (chapter.audioUrl) {
          try {
            URL.revokeObjectURL(chapter.audioUrl)
          } catch {}
        }

        // Get audio duration
        const duration = await new Promise<number>((resolve) => {
          const audio = new Audio(url)
          audio.addEventListener("loadedmetadata", () => {
            resolve(audio.duration)
          })
          audio.addEventListener("error", () => {
            resolve(0)
          })
        })

        updateChapter(chapter.id, {
          status: "completed",
          audioBlob: blob,
          audioUrl: url,
          audioDuration: duration
        })

        return true
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error"
        updateChapter(chapter.id, {
          status: "error",
          errorMessage
        })
        return false
      } finally {
        setCurrentGeneratingId(null)
      }
    },
    [generateChapterAudio, updateChapter, setCurrentGeneratingId]
  )

  const generateAllChapters = useCallback(
    async ({
      chapters,
      overrides,
      onChapterStart,
      onChapterComplete,
      onChapterError
    }: GenerateAllOptions): Promise<{ success: number; failed: number }> => {
      const pendingChapters = chapters.filter(
        (ch) => ch.status === "pending" || ch.status === "error"
      )

      if (pendingChapters.length === 0) {
        return { success: 0, failed: 0 }
      }

      abortControllerRef.current = new AbortController()
      setIsGenerating(true)
      setGenerationQueue(pendingChapters.map((ch) => ch.id))

      let success = 0
      let failed = 0

      for (const chapter of pendingChapters) {
        if (abortControllerRef.current?.signal.aborted) {
          break
        }

        onChapterStart?.(chapter.id)
        setCurrentGeneratingId(chapter.id)
        updateChapter(chapter.id, { status: "generating", errorMessage: undefined })

        try {
          const blob = await generateChapterAudio({ chapter, overrides })

          if (abortControllerRef.current?.signal.aborted) {
            updateChapter(chapter.id, { status: "pending" })
            break
          }

          if (!blob) {
            updateChapter(chapter.id, { status: "pending", errorMessage: undefined })
            continue
          }

          const url = URL.createObjectURL(blob)
          if (chapter.audioUrl) {
            try {
              URL.revokeObjectURL(chapter.audioUrl)
            } catch {}
          }

          const duration = await new Promise<number>((resolve) => {
            const audio = new Audio(url)
            audio.addEventListener("loadedmetadata", () => {
              resolve(audio.duration)
            })
            audio.addEventListener("error", () => {
              resolve(0)
            })
          })

          updateChapter(chapter.id, {
            status: "completed",
            audioBlob: blob,
            audioUrl: url,
            audioDuration: duration
          })

          success++
          onChapterComplete?.(chapter.id)
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : "Unknown error"
          updateChapter(chapter.id, {
            status: "error",
            errorMessage
          })
          failed++
          onChapterError?.(
            chapter.id,
            error instanceof Error ? error : new Error(errorMessage)
          )
        }

        // Remove from queue
        const currentQueue = getGenerationQueue()
        setGenerationQueue(currentQueue.filter((id) => id !== chapter.id))
      }

      setIsGenerating(false)
      setCurrentGeneratingId(null)
      setGenerationQueue([])
      abortControllerRef.current = null

      return { success, failed }
    },
    [
      generateChapterAudio,
      updateChapter,
      setIsGenerating,
      setCurrentGeneratingId,
      setGenerationQueue
    ]
  )

  const cancelGeneration = useCallback(() => {
    abortControllerRef.current?.abort()
    setIsGenerating(false)
    setCurrentGeneratingId(null)
    setGenerationQueue([])
  }, [setIsGenerating, setCurrentGeneratingId, setGenerationQueue])

  const downloadChapter = useCallback(
    (chapter: AudioChapter, filename?: string) => {
      if (!chapter.audioBlob || !chapter.audioUrl) {
        notification.warning({
          message: t("audiobook:noAudioToDownload", "No audio to download"),
          description: t(
            "audiobook:generateFirst",
            "Generate the chapter audio first."
          )
        })
        return
      }

      const extension = resolveAudioExtension(chapter.audioBlob.type)
      const name =
        filename ||
        `${chapter.title.replace(/[^a-zA-Z0-9]/g, "_")}.${extension}`

      const a = document.createElement("a")
      a.href = chapter.audioUrl
      a.download = name
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    },
    [notification, t]
  )

  const downloadAllChapters = useCallback(
    (chapters: AudioChapter[], projectTitle?: string) => {
      const completedChapters = chapters
        .filter((ch) => ch.status === "completed" && ch.audioBlob)
        .sort((a, b) => a.order - b.order)

      if (completedChapters.length === 0) {
        notification.warning({
          message: t("audiobook:noChaptersToDownload", "No chapters to download"),
          description: t(
            "audiobook:generateChaptersFirst",
            "Generate chapter audio first."
          )
        })
        return
      }

      const prefix = projectTitle?.replace(/[^a-zA-Z0-9]/g, "_") || "audiobook"

      completedChapters.forEach((chapter, idx) => {
        const extension = resolveAudioExtension(chapter.audioBlob?.type || "")
        const name = `${prefix}_${String(idx + 1).padStart(2, "0")}_${chapter.title.replace(/[^a-zA-Z0-9]/g, "_")}.${extension}`
        downloadChapter(chapter, name)
      })
    },
    [downloadChapter, notification, t]
  )

  return {
    generateSingleChapter,
    generateAllChapters,
    cancelGeneration,
    downloadChapter,
    downloadAllChapters
  }
}
