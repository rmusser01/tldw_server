import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { shallow } from "zustand/shallow"
import {
  DEFAULT_RAG_SETTINGS,
  type RagPresetName,
  type RagSettings,
  applyRagPreset,
  buildRagSearchRequest,
  toRagAdvancedOptions
} from "@/services/rag/unified-rag"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import type { RagResult, BatchResultGroup } from "./useRagResultsDisplay"

const TRANSIENT_KEYS = new Set<keyof RagSettings>(["query", "batch_queries"])

const normalizeSettings = (value?: Partial<RagSettings>) => ({
  ...DEFAULT_RAG_SETTINGS,
  ...(value || {})
})

const normalizeBatchResults = (payload: any): BatchResultGroup[] => {
  if (!payload) return []
  if (Array.isArray(payload)) {
    return payload
      .map((group: any) => ({
        query: String(group.query || ""),
        results: group.results || []
      }))
      .filter((group: BatchResultGroup) => group.results.length > 0)
  }
  if (typeof payload === "object") {
    return Object.entries(payload)
      .map(([query, results]) => ({
        query,
        results: Array.isArray(results) ? results : []
      }))
      .filter((group) => group.results.length > 0)
  }
  return []
}

const parseBatchQueries = (value: string) =>
  value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)

export interface UseRagSearchStateDeps {
  currentMessage?: string
  t: (key: string, fallback?: string) => string
}

export function useRagSearchState(deps: UseRagSearchStateDeps) {
  const { currentMessage, t } = deps

  const [preset, setPreset] = useStorage<RagPresetName>(
    "ragSearchPreset",
    "balanced"
  )
  const [storedSettings, setStoredSettings] = useStorage<RagSettings>(
    "ragSearchSettingsV2",
    DEFAULT_RAG_SETTINGS
  )
  const [useCurrentMessage, setUseCurrentMessage] = useStorage<boolean>(
    "ragSearchUseCurrentMessage",
    true
  )
  const [draftSettings, setDraftSettings] = React.useState<RagSettings>(
    normalizeSettings(storedSettings)
  )
  const [loading, setLoading] = React.useState(false)
  const [queryError, setQueryError] = React.useState<string | null>(null)
  const [timedOut, setTimedOut] = React.useState(false)
  const [results, setResults] = React.useState<RagResult[]>([])
  const [batchResults, setBatchResults] = React.useState<BatchResultGroup[]>([])
  const [hasAttemptedSearch, setHasAttemptedSearch] = React.useState(false)
  const [ragHintSeen, setRagHintSeen] = useStorage<boolean>(
    "ragSearchHintSeen",
    false
  )

  const {
    ragPinnedResults,
    setRagPinnedResults,
    setRagSearchMode,
    setRagTopK,
    setRagEnableGeneration,
    setRagEnableCitations,
    setRagSources,
    setRagAdvancedOptions
  } = useStoreMessageOption(
    (state) => ({
      ragPinnedResults: state.ragPinnedResults,
      setRagPinnedResults: state.setRagPinnedResults,
      setRagSearchMode: state.setRagSearchMode,
      setRagTopK: state.setRagTopK,
      setRagEnableGeneration: state.setRagEnableGeneration,
      setRagEnableCitations: state.setRagEnableCitations,
      setRagSources: state.setRagSources,
      setRagAdvancedOptions: state.setRagAdvancedOptions
    }),
    shallow
  )

  React.useEffect(() => {
    setDraftSettings(normalizeSettings(storedSettings))
  }, [storedSettings])

  const updateSetting = React.useCallback(
    <K extends keyof RagSettings>(
      key: K,
      value: RagSettings[K],
      options?: { transient?: boolean }
    ) => {
      setDraftSettings((prev) => ({
        ...prev,
        [key]: value
      }))
      if (!options?.transient && preset !== "custom") {
        if (!TRANSIENT_KEYS.has(key)) {
          setPreset("custom")
        }
      }
    },
    [preset, setPreset]
  )

  const applyPresetSelection = React.useCallback(
    (nextPreset: RagPresetName) => {
      setPreset(nextPreset)
      if (nextPreset === "custom") return
      const nextSettings = applyRagPreset(nextPreset)
      nextSettings.query = draftSettings.query
      nextSettings.batch_queries = draftSettings.batch_queries
      setDraftSettings(nextSettings)
    },
    [draftSettings.batch_queries, draftSettings.query, setPreset]
  )

  const applySettings = React.useCallback(() => {
    const persistedSettings = {
      ...draftSettings,
      query: "",
      batch_queries: []
    }
    setStoredSettings(persistedSettings)
    setRagSearchMode(draftSettings.search_mode)
    setRagTopK(draftSettings.top_k)
    setRagEnableGeneration(draftSettings.enable_generation)
    setRagEnableCitations(draftSettings.enable_citations)
    setRagSources(draftSettings.sources)
    setRagAdvancedOptions(toRagAdvancedOptions(draftSettings))
  }, [
    draftSettings,
    setRagEnableCitations,
    setRagEnableGeneration,
    setRagSearchMode,
    setRagSources,
    setRagTopK,
    setRagAdvancedOptions,
    setStoredSettings
  ])

  const resolvedQuery = React.useMemo(() => {
    if (useCurrentMessage && !draftSettings.query.trim()) {
      return (currentMessage || "").trim()
    }
    return draftSettings.query.trim()
  }, [currentMessage, draftSettings.query, useCurrentMessage])

  const resetToBalanced = () => {
    applyPresetSelection("balanced")
  }

  const runSearch = async (opts?: { applyFirst?: boolean }) => {
    if (opts?.applyFirst) {
      applySettings()
    }
    const hasBatchQueries =
      draftSettings.enable_batch && draftSettings.batch_queries.length > 0
    const query = resolvedQuery || (hasBatchQueries ? draftSettings.batch_queries[0] : "")
    if (!query) {
      setQueryError(
        t("sidepanel:rag.queryRequired", "Enter a query to search.") as string
      )
      return
    }
    setQueryError(null)
    if (!hasAttemptedSearch) {
      setHasAttemptedSearch(true)
      setRagHintSeen(true)
    }
    setLoading(true)
    setTimedOut(false)
    setResults([])
    setBatchResults([])
    try {
      await tldwClient.initialize()
      const settings = {
        ...draftSettings,
        query
      }
      const { query: resolved, options, timeoutMs } =
        buildRagSearchRequest(settings)
      const ragRes = await tldwClient.ragSearch(resolved, {
        ...options,
        timeoutMs
      })
      const grouped = normalizeBatchResults(
        ragRes?.batch_results || ragRes?.results_by_query
      )
      if (grouped.length > 0) {
        setBatchResults(grouped)
      } else {
        const docs = ragRes?.results || ragRes?.documents || ragRes?.docs || []
        setResults(docs)
      }
      setTimedOut(false)
    } catch (e) {
      setResults([])
      setBatchResults([])
      setTimedOut(true)
    } finally {
      setLoading(false)
    }
  }

  return {
    // storage-backed state
    preset,
    storedSettings,
    useCurrentMessage,
    setUseCurrentMessage,
    ragHintSeen,
    setRagHintSeen,
    hasAttemptedSearch,
    // draft settings
    draftSettings,
    setDraftSettings,
    // search state
    loading,
    queryError,
    timedOut,
    results,
    batchResults,
    resolvedQuery,
    // store
    ragPinnedResults,
    setRagPinnedResults,
    // callbacks
    updateSetting,
    applyPresetSelection,
    applySettings,
    resetToBalanced,
    runSearch,
    normalizeSettings,
    parseBatchQueries,
  }
}
