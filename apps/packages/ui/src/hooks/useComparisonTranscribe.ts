import { useCallback, useEffect, useRef, useState } from "react"

import {
  classifyAudioError,
  type AudioErrorCategory
} from "@/components/Option/Audio/audio-error-classification"
import {
  buildSttComparisonConfig,
  normalizeSttResponse,
  type SttComparisonConfig,
  type SttComparisonMetadata
} from "@/components/Option/Audio/comparison-provenance"
import { resolveServicePromptScope } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export interface ComparisonResult {
  id: string
  model: string
  text: string
  status: "pending" | "running" | "done" | "error"
  disabled?: boolean
  config?: SttComparisonConfig
  metadata?: SttComparisonMetadata
  error?: string
  errorCategory?: AudioErrorCategory
  errorRecovery?: string
  errorSettingsHref?: "/settings/speech"
  errorDebugMessage?: string
  latencyMs?: number
  wordCount?: number
  requestOptions?: Record<string, unknown>
}

function countWords(text: string): number {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean).length
}

function cloneRequestOptions(
  options?: Record<string, unknown>
): Record<string, unknown> {
  return { ...(options || {}) }
}

function cloneConfig(config?: SttComparisonConfig): SttComparisonConfig | undefined {
  if (!config) return undefined
  return {
    ...config,
    timestampGranularities: config.timestampGranularities
      ? [...config.timestampGranularities]
      : undefined
  }
}

export function useComparisonTranscribe(isCurrentOwner: () => boolean) {
  const [results, setResults] = useState<ComparisonResult[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const resultsRef = useRef<ComparisonResult[]>([])
  const nextResultId = useRef(0)
  const epoch = useRef(0)
  const controllers = useRef(new Set<AbortController>())
  const retire = useCallback(() => {
    epoch.current++
    for (const controller of controllers.current) controller.abort()
    controllers.current.clear()
  }, [])
  useEffect(() => retire, [retire])

  const createResultId = useCallback((model: string) => {
    nextResultId.current += 1
    return `${model}-${nextResultId.current}`
  }, [])

  const updateResult = useCallback((id: string, patch: Partial<ComparisonResult>) => {
    setResults((prev) => {
      const next = prev.map((r) => (r.id === id ? { ...r, ...patch } : r))
      resultsRef.current = next
      return next
    })
  }, [])

  const runSingleTranscription = useCallback(
    async (
      blob: Blob,
      resultId: string,
      model: string,
      sttOptions: Record<string, unknown>
    ) => {
      if (!isCurrentOwner()) return
      const lease = epoch.current
      const controller = new AbortController()
      controllers.current.add(controller)
      setIsRunning(true)
      const current = () => isCurrentOwner() && epoch.current === lease && !controller.signal.aborted
      const requestOptions = { ...sttOptions }
      const config = buildSttComparisonConfig(model, requestOptions)
      const createdAt = new Date().toISOString()
      const baseMetadata: SttComparisonMetadata = {
        createdAt,
        audioSourceLabel: typeof File !== "undefined" && blob instanceof File ? blob.name : "Recorded audio",
        audioSizeBytes: blob.size
      }
      updateResult(resultId, {
        status: "running",
        text: "",
        config,
        metadata: baseMetadata,
        requestOptions,
        error: undefined,
        errorCategory: undefined,
        errorRecovery: undefined,
        errorSettingsHref: undefined,
        errorDebugMessage: undefined,
        latencyMs: undefined,
        wordCount: undefined
      })
      const start = performance.now()
      try {
        const requestScope = await resolveServicePromptScope()
        if (!current()) return
        const response = await tldwClient.transcribeAudio(blob, {
          ...requestOptions,
          model, signal: controller.signal, requestScope
        })
        if (!current()) return
        const normalized = normalizeSttResponse(response)
        const text = normalized.text
        const latencyMs = performance.now() - start
        const wordCount = normalized.metadata.wordCount ?? countWords(text)
        updateResult(resultId, {
          status: "done",
          text,
          latencyMs,
          wordCount,
          error: undefined,
          errorCategory: undefined,
          errorRecovery: undefined,
          errorSettingsHref: undefined,
          errorDebugMessage: undefined,
          metadata: {
            ...baseMetadata,
            ...normalized.metadata,
            clientLatencyMs: latencyMs,
            wordCount
          }
        })
      } catch (err: unknown) {
        if (!current()) return
        const classified = classifyAudioError(err)
        const latencyMs = performance.now() - start
        updateResult(resultId, {
          status: "error",
          error: classified.title,
          errorCategory: classified.category,
          errorRecovery: classified.recovery,
          errorSettingsHref: classified.settingsHref,
          errorDebugMessage: classified.debugMessage,
          text: "",
          latencyMs,
          metadata: {
            ...baseMetadata,
            clientLatencyMs: latencyMs,
            errorCategory: classified.category
          }
        })
      } finally {
        controllers.current.delete(controller)
        if (epoch.current === lease) setIsRunning(controllers.current.size > 0)
      }
    },
    [updateResult, isCurrentOwner]
  )

  const transcribeAll = useCallback(
    async (blob: Blob, models: string[], sttOptions: Record<string, unknown>) => {
      if (!isCurrentOwner()) return
      const existing = resultsRef.current
      const existingForSelectedModels = existing.filter((result) =>
        models.includes(result.model)
      )
      const modelsWithoutRows = models.filter(
        (model) => !existing.some((result) => result.model === model)
      )
      const newRows: ComparisonResult[] = modelsWithoutRows.map((model) => ({
        id: createResultId(model),
        model,
        text: "",
        status: "pending" as const,
        config: buildSttComparisonConfig(model, sttOptions),
        metadata: {
          createdAt: new Date().toISOString(),
          audioSourceLabel: typeof File !== "undefined" && blob instanceof File ? blob.name : "Recorded audio",
          audioSizeBytes: blob.size
        },
        requestOptions: cloneRequestOptions(sttOptions)
      }))
      const nextResults =
        existing.length > 0 ? [...existing, ...newRows] : newRows
      const rowsToRun =
        existing.length > 0
          ? [...existingForSelectedModels, ...newRows].filter(
              (result) => !result.disabled
            )
          : nextResults

      resultsRef.current = nextResults
      setResults(nextResults)
      if (rowsToRun.length === 0) return
      setIsRunning(true)
      const lease = epoch.current

      try {
        await Promise.allSettled(
          rowsToRun.map((result) =>
            runSingleTranscription(
              blob,
              result.id,
              result.model,
              result.requestOptions || sttOptions
            )
          )
        )
      } finally {
        if (epoch.current === lease) setIsRunning(controllers.current.size > 0)
      }
    },
    [createResultId, runSingleTranscription, isCurrentOwner]
  )

  const retryModel = useCallback(
    async (blob: Blob, target: string, sttOptions: Record<string, unknown>) => {
      const existing = resultsRef.current.find(
        (r) => r.id === target || r.model === target
      )
      if (!existing) return

      await runSingleTranscription(
        blob,
        existing.id,
        existing.model,
        existing.requestOptions || sttOptions
      )
    },
    [runSingleTranscription]
  )

  const duplicateResult = useCallback(
    (target: string) => {
      setResults((prev) => {
        const source = prev.find((r) => r.id === target || r.model === target)
        if (!source) return prev

        const duplicate: ComparisonResult = {
          id: createResultId(source.model),
          model: source.model,
          text: "",
          status: "pending",
          disabled: false,
          config: cloneConfig(source.config),
          metadata: source.metadata
            ? {
                ...source.metadata,
                createdAt: new Date().toISOString(),
                clientLatencyMs: undefined,
                errorCategory: undefined
              }
            : undefined,
          requestOptions: cloneRequestOptions(source.requestOptions)
        }
        const next = [...prev, duplicate]
        resultsRef.current = next
        return next
      })
    },
    [createResultId]
  )

  const setResultDisabled = useCallback((target: string, disabled: boolean) => {
    setResults((prev) => {
      const next = prev.map((result) =>
        result.id === target || result.model === target
          ? { ...result, disabled }
          : result
      )
      resultsRef.current = next
      return next
    })
  }, [])

  const clearResults = useCallback(() => {
    retire()
    resultsRef.current = []
    setResults([])
    setIsRunning(false)
  }, [retire])

  return {
    results,
    isRunning,
    transcribeAll,
    retryModel,
    duplicateResult,
    setResultDisabled,
    clearResults
  }
}
