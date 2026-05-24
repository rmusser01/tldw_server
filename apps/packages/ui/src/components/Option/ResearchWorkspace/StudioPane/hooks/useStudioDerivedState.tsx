import React, { useState, useEffect, useRef, useMemo } from "react"
import type { AudioTtsProvider } from "@/types/workspace"

const OUTPUT_VIRTUALIZATION_THRESHOLD = 50
const OUTPUT_VIRTUAL_ROW_HEIGHT = 150
const OUTPUT_VIRTUAL_OVERSCAN = 4
const STUDIO_DEFAULT_RAG_TOP_K = 8
const STUDIO_DEFAULT_RAG_MIN_SCORE = 0.2
const STUDIO_DEFAULT_ENABLE_RERANKING = true
const STUDIO_DEFAULT_MAX_TOKENS = 800
const STUDIO_DEFAULT_SUMMARY_INSTRUCTION =
  "Provide a comprehensive summary of the key points and main ideas."

export {
  STUDIO_DEFAULT_RAG_TOP_K,
  STUDIO_DEFAULT_RAG_MIN_SCORE,
  STUDIO_DEFAULT_ENABLE_RERANKING,
  STUDIO_DEFAULT_MAX_TOKENS,
  STUDIO_DEFAULT_SUMMARY_INSTRUCTION,
  OUTPUT_VIRTUALIZATION_THRESHOLD,
  OUTPUT_VIRTUAL_ROW_HEIGHT,
  OUTPUT_VIRTUAL_OVERSCAN
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

export interface UseStudioDerivedStateDeps {
  ragTopK: number | undefined
  ragAdvancedOptions: unknown
  temperature: number | undefined
  topP: number | undefined
  numPredict: number | undefined
  setRagTopK: (v: number) => void
  setRagAdvancedOptions: (v: any) => void
  setTemperature: (v: any) => void
  setTopP: (v: any) => void
  generatedArtifacts: any[]
  outputListScrollTop: number
  outputListViewportHeight: number
}

export function useStudioDerivedState(deps: UseStudioDerivedStateDeps) {
  const {
    ragTopK,
    ragAdvancedOptions,
    temperature,
    topP,
    numPredict,
    setRagTopK,
    setRagAdvancedOptions,
    setTemperature,
    setTopP,
    generatedArtifacts,
    outputListScrollTop,
    outputListViewportHeight
  } = deps

  const normalizedRagAdvancedOptions = React.useMemo(() => {
    return isRecord(ragAdvancedOptions) ? ragAdvancedOptions : {}
  }, [ragAdvancedOptions])

  const resolvedSummaryInstruction = React.useMemo(() => {
    const raw = normalizedRagAdvancedOptions.generation_prompt
    return typeof raw === "string" && raw.trim().length > 0
      ? raw.trim()
      : STUDIO_DEFAULT_SUMMARY_INSTRUCTION
  }, [normalizedRagAdvancedOptions.generation_prompt])

  const resolvedStudioTopK = React.useMemo(() => {
    const value =
      typeof ragTopK === "number" && Number.isFinite(ragTopK)
        ? ragTopK
        : STUDIO_DEFAULT_RAG_TOP_K
    return Math.max(1, Math.min(50, Math.round(value)))
  }, [ragTopK])

  const studioSimilarityThreshold = React.useMemo(() => {
    const raw = normalizedRagAdvancedOptions.min_score
    const value =
      typeof raw === "number" && Number.isFinite(raw)
        ? raw
        : STUDIO_DEFAULT_RAG_MIN_SCORE
    return Math.max(0, Math.min(1, value))
  }, [normalizedRagAdvancedOptions.min_score])

  const studioRerankingEnabled = React.useMemo(() => {
    const raw = normalizedRagAdvancedOptions.enable_reranking
    return typeof raw === "boolean"
      ? raw
      : STUDIO_DEFAULT_ENABLE_RERANKING
  }, [normalizedRagAdvancedOptions.enable_reranking])

  const resolvedTemperature = React.useMemo(() => {
    const value =
      typeof temperature === "number" && Number.isFinite(temperature)
        ? temperature
        : 0.7
    return Math.max(0, Math.min(2, Number(value.toFixed(2))))
  }, [temperature])

  const resolvedTopP = React.useMemo(() => {
    const value = typeof topP === "number" && Number.isFinite(topP) ? topP : 1
    return Math.max(0, Math.min(1, Number(value.toFixed(2))))
  }, [topP])

  const resolvedNumPredict = React.useMemo(() => {
    const value =
      typeof numPredict === "number" && Number.isFinite(numPredict)
        ? numPredict
        : STUDIO_DEFAULT_MAX_TOKENS
    return Math.max(1, Math.min(32768, Math.round(value)))
  }, [numPredict])

  // Local handlers
  const patchRagAdvancedOptions = React.useCallback(
    (patch: Record<string, unknown>) => {
      setRagAdvancedOptions({
        ...normalizedRagAdvancedOptions,
        ...patch
      })
    },
    [normalizedRagAdvancedOptions, setRagAdvancedOptions]
  )

  const handleStudioTopKChange = React.useCallback((value: number | number[]) => {
    const raw = Array.isArray(value) ? value[0] : value
    if (typeof raw !== "number" || !Number.isFinite(raw)) return
    const nextTopK = Math.max(1, Math.min(50, Math.round(raw)))
    setRagTopK(nextTopK)
    patchRagAdvancedOptions({ top_k: nextTopK })
  }, [setRagTopK, patchRagAdvancedOptions])

  const handleStudioSimilarityThresholdChange = React.useCallback((value: number | number[]) => {
    const raw = Array.isArray(value) ? value[0] : value
    if (typeof raw !== "number" || !Number.isFinite(raw)) return
    const nextThreshold = Math.max(0, Math.min(1, raw))
    patchRagAdvancedOptions({ min_score: Number(nextThreshold.toFixed(2)) })
  }, [patchRagAdvancedOptions])

  const handleStudioTemperatureChange = React.useCallback((value: number | number[]) => {
    const raw = Array.isArray(value) ? value[0] : value
    if (typeof raw !== "number" || !Number.isFinite(raw)) return
    setTemperature(Math.max(0, Math.min(2, Number(raw.toFixed(2)))))
  }, [setTemperature])

  const handleStudioTopPChange = React.useCallback((value: number | number[]) => {
    const raw = Array.isArray(value) ? value[0] : value
    if (typeof raw !== "number" || !Number.isFinite(raw)) return
    setTopP(Math.max(0, Math.min(1, Number(raw.toFixed(2)))))
  }, [setTopP])

  // Virtualization
  const useVirtualizedOutputs =
    generatedArtifacts.length > OUTPUT_VIRTUALIZATION_THRESHOLD
  const virtualOutputStartIndex = useVirtualizedOutputs
    ? Math.max(
        0,
        Math.floor(outputListScrollTop / OUTPUT_VIRTUAL_ROW_HEIGHT) -
          OUTPUT_VIRTUAL_OVERSCAN
      )
    : 0
  const virtualOutputEndIndex = useVirtualizedOutputs
    ? Math.min(
        generatedArtifacts.length,
        Math.ceil(
          (outputListScrollTop + outputListViewportHeight) /
            OUTPUT_VIRTUAL_ROW_HEIGHT
        ) + OUTPUT_VIRTUAL_OVERSCAN
      )
    : generatedArtifacts.length
  const visibleArtifacts = useVirtualizedOutputs
    ? generatedArtifacts.slice(virtualOutputStartIndex, virtualOutputEndIndex)
    : generatedArtifacts
  const virtualOutputTopPadding = useVirtualizedOutputs
    ? virtualOutputStartIndex * OUTPUT_VIRTUAL_ROW_HEIGHT
    : 0
  const virtualOutputBottomPadding = useVirtualizedOutputs
    ? Math.max(
        0,
        (generatedArtifacts.length - virtualOutputEndIndex) *
          OUTPUT_VIRTUAL_ROW_HEIGHT
      )
    : 0

  return {
    normalizedRagAdvancedOptions,
    resolvedSummaryInstruction,
    resolvedStudioTopK,
    studioSimilarityThreshold,
    studioRerankingEnabled,
    resolvedTemperature,
    resolvedTopP,
    resolvedNumPredict,
    patchRagAdvancedOptions,
    handleStudioTopKChange,
    handleStudioSimilarityThresholdChange,
    handleStudioTemperatureChange,
    handleStudioTopPChange,
    useVirtualizedOutputs,
    visibleArtifacts,
    virtualOutputTopPadding,
    virtualOutputBottomPadding
  }
}
