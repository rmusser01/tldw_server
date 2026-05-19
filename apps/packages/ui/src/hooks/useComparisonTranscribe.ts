import { useCallback, useRef, useState } from "react"

import {
  classifyAudioError,
  type AudioErrorCategory
} from "@/components/Option/Audio/audio-error-classification"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export interface ComparisonResult {
  model: string
  text: string
  status: "pending" | "running" | "done" | "error"
  error?: string
  errorCategory?: AudioErrorCategory
  errorRecovery?: string
  errorSettingsHref?: "/settings/speech"
  errorDebugMessage?: string
  latencyMs?: number
  wordCount?: number
}

/**
 * Extract text from various API response shapes:
 * - string
 * - { text: string }
 * - { transcript: string }
 * - { segments: [{ text: string }] }
 */
function extractText(response: unknown): string {
  if (typeof response === "string") return response
  if (response && typeof response === "object") {
    const obj = response as Record<string, unknown>
    if (typeof obj.text === "string") return obj.text
    if (typeof obj.transcript === "string") return obj.transcript
    if (Array.isArray(obj.segments)) {
      return obj.segments
        .map((s: Record<string, unknown>) => (typeof s.text === "string" ? s.text : ""))
        .join(" ")
    }
  }
  return ""
}

function countWords(text: string): number {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean).length
}

export function useComparisonTranscribe() {
  const [results, setResults] = useState<ComparisonResult[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const resultsRef = useRef<ComparisonResult[]>([])

  const updateResult = useCallback((model: string, patch: Partial<ComparisonResult>) => {
    setResults((prev) => {
      const next = prev.map((r) => (r.model === model ? { ...r, ...patch } : r))
      resultsRef.current = next
      return next
    })
  }, [])

  const runSingleTranscription = useCallback(
    async (blob: Blob, model: string, sttOptions: Record<string, unknown>) => {
      updateResult(model, { status: "running" })
      const start = performance.now()
      try {
        const response = await tldwClient.transcribeAudio(blob, { ...sttOptions, model })
        const text = extractText(response)
        const latencyMs = performance.now() - start
        updateResult(model, {
          status: "done",
          text,
          latencyMs,
          wordCount: countWords(text)
        })
      } catch (err: unknown) {
        const classified = classifyAudioError(err)
        updateResult(model, {
          status: "error",
          error: classified.title,
          errorCategory: classified.category,
          errorRecovery: classified.recovery,
          errorSettingsHref: classified.settingsHref,
          errorDebugMessage: classified.debugMessage,
          text: "",
          latencyMs: performance.now() - start
        })
      }
    },
    [updateResult]
  )

  const transcribeAll = useCallback(
    async (blob: Blob, models: string[], sttOptions: Record<string, unknown>) => {
      const initial: ComparisonResult[] = models.map((model) => ({
        model,
        text: "",
        status: "pending" as const
      }))
      resultsRef.current = initial
      setResults(initial)
      setIsRunning(true)

      await Promise.allSettled(
        models.map((model) => runSingleTranscription(blob, model, sttOptions))
      )

      setIsRunning(false)
    },
    [runSingleTranscription]
  )

  const retryModel = useCallback(
    async (blob: Blob, model: string, sttOptions: Record<string, unknown>) => {
      const exists = resultsRef.current.some((r) => r.model === model)
      if (!exists) return

      await runSingleTranscription(blob, model, sttOptions)
    },
    [runSingleTranscription]
  )

  const clearResults = useCallback(() => {
    resultsRef.current = []
    setResults([])
    setIsRunning(false)
  }, [])

  return { results, isRunning, transcribeAll, retryModel, clearResults }
}
