/**
 * Hook: useWritingInspectorPanels
 *
 * Manages inspector panel state: token inspector, response inspector (logprobs),
 * wordcloud generation, and panel toggling.
 */

import React from "react"
import type { TFunction } from "i18next"
import {
  countWritingTokens,
  createWritingWordcloud,
  getWritingWordcloud,
  tokenizeWritingText,
  type WritingTokenCountResponse,
  type WritingTokenizeResponse,
  type WritingWordcloudResponse
} from "@/services/writing-playground"
import {
  AUTO_MODEL_ID,
  resolveApiProviderForModel
} from "@/utils/resolve-api-provider"
import {
  extractLogprobEntriesFromChunk,
  type WritingLogprobEntry
} from "../writing-logprob-utils"
import { buildTokenPreviewRows, joinTokenStrings } from "../writing-token-utils"
import {
  buildRerollPromptFromRows,
  buildResponseInspectorCsv,
  normalizeInspectorToken,
  selectResponseInspectorRows,
  type ResponseInspectorSort
} from "../writing-response-inspector-utils"
import {
  normalizeWordcloudWords,
  parseWordcloudStopwordsInput
} from "../writing-wordcloud-utils"
import type { InspectorTabKey } from "../WritingPlayground.types"
import {
  MAX_RESPONSE_LOGPROBS,
  PREDICT_PLACEHOLDER,
  wait,
  WORDCLOUD_POLL_ATTEMPTS,
  WORDCLOUD_POLL_DELAY_MS,
  type LastGenerationContext
} from "./utils"

export interface UseWritingInspectorPanelsDeps {
  isOnline: boolean
  selectedModel: string | undefined
  apiProviderOverride: string | undefined
  activeSessionId: string | null
  activeSessionDetail: unknown | null
  settingsDisabled: boolean
  isGenerating: boolean
  writingCaps: { server?: Record<string, unknown> } | undefined
  requestedCaps: { requested?: Record<string, unknown> } | undefined
  requestedCapsLoading: boolean
  requestedLogprobsExplicitlyUnsupported: boolean
  requestedLogprobsSupported: boolean
  requestedTopLogprobsSupported: boolean
  settings: { logprobs: boolean; top_logprobs: number | null }
  lastGenerationContextRef: React.MutableRefObject<LastGenerationContext | null>
  handleGenerate: (overrideText?: string) => void
  t: TFunction
}

export function useWritingInspectorPanels(deps: UseWritingInspectorPanelsDeps) {
  const {
    isOnline,
    selectedModel,
    apiProviderOverride,
    activeSessionId,
    settingsDisabled,
    isGenerating,
    writingCaps,
    requestedCaps,
    requestedCapsLoading,
    requestedLogprobsExplicitlyUnsupported,
    requestedLogprobsSupported,
    requestedTopLogprobsSupported,
    settings,
    lastGenerationContextRef,
    handleGenerate,
    t
  } = deps

  // --- Panel state ---
  const [libraryOpen, setLibraryOpen] = React.useState(true)
  const [inspectorOpen, setInspectorOpen] = React.useState(false)
  const [activeInspectorTab, setActiveInspectorTab] =
    React.useState<InspectorTabKey>("sampling")

  // --- Token inspector ---
  const [tokenCountResult, setTokenCountResult] =
    React.useState<WritingTokenCountResponse | null>(null)
  const [tokenizeResult, setTokenizeResult] =
    React.useState<WritingTokenizeResponse | null>(null)
  const [tokenInspectorError, setTokenInspectorError] =
    React.useState<string | null>(null)
  const [isCountingTokens, setIsCountingTokens] = React.useState(false)
  const [isTokenizingText, setIsTokenizingText] = React.useState(false)

  // --- Response inspector ---
  const [responseLogprobs, setResponseLogprobs] =
    React.useState<WritingLogprobEntry[]>([])
  const [responseInspectorQuery, setResponseInspectorQuery] = React.useState("")
  const [responseInspectorSort, setResponseInspectorSort] =
    React.useState<ResponseInspectorSort>("sequence")
  const [responseInspectorHideWhitespace, setResponseInspectorHideWhitespace] =
    React.useState(false)
  const [generationTokenCount, setGenerationTokenCount] = React.useState(0)
  const [generationTokensPerSec, setGenerationTokensPerSec] = React.useState(0)

  // --- Wordcloud ---
  const [wordcloudId, setWordcloudId] = React.useState<string | null>(null)
  const [wordcloudStatus, setWordcloudStatus] = React.useState<string | null>(null)
  const [wordcloudWords, setWordcloudWords] =
    React.useState<Array<{ text: string; weight: number }>>([])
  const [wordcloudMeta, setWordcloudMeta] = React.useState<{
    input_chars: number
    total_tokens: number
    top_n: number
  } | null>(null)
  const [wordcloudError, setWordcloudError] = React.useState<string | null>(null)
  const [isGeneratingWordcloud, setIsGeneratingWordcloud] = React.useState(false)
  const [wordcloudMaxWords, setWordcloudMaxWords] = React.useState(100)
  const [wordcloudMinWordLength, setWordcloudMinWordLength] = React.useState(3)
  const [wordcloudKeepNumbers, setWordcloudKeepNumbers] = React.useState(false)
  const [wordcloudStopwordsInput, setWordcloudStopwordsInput] = React.useState("")
  const wordcloudRequestSeqRef = React.useRef(0)
  const wordcloudUnmountedRef = React.useRef(false)

  // --- Generation timer ---
  const [generationElapsed, setGenerationElapsed] = React.useState(0)
  const generationTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(null)
  const [showPromptChunks, setShowPromptChunks] = React.useState(false)

  // --- Effects ---
  React.useEffect(() => {
    if (isGenerating) {
      setGenerationElapsed(0)
      const start = Date.now()
      generationTimerRef.current = setInterval(() => {
        setGenerationElapsed(Math.floor((Date.now() - start) / 1000))
      }, 1000)
    } else {
      if (generationTimerRef.current) {
        clearInterval(generationTimerRef.current)
        generationTimerRef.current = null
      }
    }
    return () => {
      if (generationTimerRef.current) {
        clearInterval(generationTimerRef.current)
        generationTimerRef.current = null
      }
    }
  }, [isGenerating])

  React.useEffect(() => {
    wordcloudUnmountedRef.current = false
    return () => {
      wordcloudUnmountedRef.current = true
    }
  }, [])

  // Reset inspector state on session change
  React.useEffect(() => {
    setResponseLogprobs([])
    setResponseInspectorQuery("")
    setResponseInspectorSort("sequence")
    setResponseInspectorHideWhitespace(false)
    setGenerationTokenCount(0)
    setGenerationTokensPerSec(0)
    setWordcloudId(null)
    setWordcloudStatus(null)
    setWordcloudWords([])
    setWordcloudMeta(null)
    setWordcloudError(null)
    setIsGeneratingWordcloud(false)
  }, [activeSessionId])

  // Reset on model change
  React.useEffect(() => {
    setTokenCountResult(null)
    setTokenizeResult(null)
    setTokenInspectorError(null)
    setResponseLogprobs([])
    setResponseInspectorQuery("")
    setResponseInspectorSort("sequence")
    setResponseInspectorHideWhitespace(false)
    setGenerationTokenCount(0)
    setGenerationTokensPerSec(0)
    setWordcloudId(null)
    setWordcloudStatus(null)
    setWordcloudWords([])
    setWordcloudMeta(null)
    setWordcloudError(null)
    setIsGeneratingWordcloud(false)
  }, [selectedModel])

  // --- Token inspector ---
  const resolveTokenInspectorTarget = React.useCallback(async () => {
    const model = String(selectedModel || "").trim()
    if (!model) {
      throw new Error(
        t("option:writingPlayground.modelMissing", "Select a model in Settings to generate.")
      )
    }
    if (model.toLowerCase() === AUTO_MODEL_ID) {
      throw new Error(
        t(
          "option:writingPlayground.tokenInspectorAutoUnavailable",
          "Token inspection requires a concrete model when Auto routing is enabled."
        )
      )
    }
    const requestedProvider = String(
      requestedCaps?.requested?.provider || ""
    ).trim()
    if (requestedProvider) {
      return { provider: requestedProvider, model }
    }
    const provider = await resolveApiProviderForModel({
      modelId: model,
      explicitProvider: apiProviderOverride
    })
    if (!provider) {
      throw new Error(
        t(
          "option:writingPlayground.tokenInspectorProviderMissing",
          "Unable to resolve a provider for the selected model."
        )
      )
    }
    return { provider, model }
  }, [apiProviderOverride, requestedCaps?.requested?.provider, selectedModel, t])

  const handleCountTokens = React.useCallback(async (editorText: string) => {
    if (!editorText.trim()) return
    setIsCountingTokens(true)
    setTokenInspectorError(null)
    try {
      const target = await resolveTokenInspectorTarget()
      const result = await countWritingTokens({
        provider: target.provider,
        model: target.model,
        text: editorText
      })
      setTokenCountResult(result)
    } catch (error) {
      const detail = error instanceof Error ? error.message : t("option:error", "Error")
      setTokenInspectorError(detail)
    } finally {
      setIsCountingTokens(false)
    }
  }, [resolveTokenInspectorTarget, t])

  const handleTokenizePreview = React.useCallback(async (editorText: string) => {
    if (!editorText.trim()) return
    setIsTokenizingText(true)
    setTokenInspectorError(null)
    try {
      const target = await resolveTokenInspectorTarget()
      const result = await tokenizeWritingText({
        provider: target.provider,
        model: target.model,
        text: editorText,
        options: { include_strings: true }
      })
      setTokenizeResult(result)
      setTokenCountResult({
        count: result.meta.token_count,
        meta: result.meta
      })
    } catch (error) {
      const detail = error instanceof Error ? error.message : t("option:error", "Error")
      setTokenInspectorError(detail)
    } finally {
      setIsTokenizingText(false)
    }
  }, [resolveTokenInspectorTarget, t])

  const clearTokenInspector = React.useCallback(() => {
    setTokenCountResult(null)
    setTokenizeResult(null)
    setTokenInspectorError(null)
  }, [])

  const clearResponseInspector = React.useCallback(() => {
    setResponseLogprobs([])
    setResponseInspectorQuery("")
    setResponseInspectorSort("sequence")
    setResponseInspectorHideWhitespace(false)
    lastGenerationContextRef.current = null
  }, [lastGenerationContextRef])

  // --- Wordcloud ---
  const applyWordcloudResponse = React.useCallback(
    (response: WritingWordcloudResponse, sequence: number) => {
      if (wordcloudUnmountedRef.current) return
      if (sequence !== wordcloudRequestSeqRef.current) return
      setWordcloudId(response.id || null)
      setWordcloudStatus(response.status || null)
      setWordcloudError(response.error?.trim() || null)
      setWordcloudWords(
        normalizeWordcloudWords(response.result?.words, wordcloudMaxWords)
      )
      setWordcloudMeta(response.result?.meta ?? null)
    },
    [wordcloudMaxWords]
  )

  const clearWordcloud = React.useCallback(() => {
    wordcloudRequestSeqRef.current += 1
    setWordcloudId(null)
    setWordcloudStatus(null)
    setWordcloudWords([])
    setWordcloudMeta(null)
    setWordcloudError(null)
    setIsGeneratingWordcloud(false)
  }, [])

  const handleGenerateWordcloud = React.useCallback(async (editorText: string) => {
    const text = editorText.trim()
    if (!text) return
    const sequence = wordcloudRequestSeqRef.current + 1
    wordcloudRequestSeqRef.current = sequence
    setIsGeneratingWordcloud(true)
    setWordcloudId(null)
    setWordcloudStatus(null)
    setWordcloudWords([])
    setWordcloudMeta(null)
    setWordcloudError(null)
    try {
      const stopwords = parseWordcloudStopwordsInput(wordcloudStopwordsInput)
      const created = await createWritingWordcloud({
        text,
        options: {
          max_words: wordcloudMaxWords,
          min_word_length: wordcloudMinWordLength,
          keep_numbers: wordcloudKeepNumbers,
          stopwords: stopwords.length > 0 ? stopwords : undefined
        }
      })
      applyWordcloudResponse(created, sequence)
      let latest = created
      if (latest.id && (latest.status === "queued" || latest.status === "running")) {
        for (let attempt = 0; attempt < WORDCLOUD_POLL_ATTEMPTS; attempt += 1) {
          if (wordcloudUnmountedRef.current) return
          if (sequence !== wordcloudRequestSeqRef.current) return
          await wait(WORDCLOUD_POLL_DELAY_MS)
          latest = await getWritingWordcloud(latest.id)
          applyWordcloudResponse(latest, sequence)
          if (latest.status === "ready" || latest.status === "failed") {
            break
          }
        }
      }
      if (
        sequence === wordcloudRequestSeqRef.current &&
        latest.status !== "ready" &&
        latest.status !== "failed"
      ) {
        setWordcloudError(
          t(
            "option:writingPlayground.wordcloudTimeout",
            "Wordcloud generation is taking longer than expected. Try again in a moment."
          )
        )
      }
    } catch (error) {
      if (wordcloudUnmountedRef.current) return
      if (sequence !== wordcloudRequestSeqRef.current) return
      const detail = error instanceof Error ? error.message : t("option:error", "Error")
      setWordcloudError(detail)
    } finally {
      if (wordcloudUnmountedRef.current) return
      if (sequence !== wordcloudRequestSeqRef.current) return
      setIsGeneratingWordcloud(false)
    }
  }, [
    applyWordcloudResponse,
    wordcloudKeepNumbers,
    wordcloudMaxWords,
    wordcloudMinWordLength,
    wordcloudStopwordsInput,
    t
  ])

  // --- Response inspector derived ---
  const handleCopyResponseInspectorJson = React.useCallback(async () => {
    if (responseLogprobs.length === 0) return
    const payload = responseLogprobs.map((entry) => ({
      token: entry.token,
      logprob: entry.logprob,
      probability: Math.exp(entry.logprob),
      top_logprobs: entry.topLogprobs
    }))
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
    } catch {
      // handled by caller
    }
  }, [responseLogprobs])

  const handleRerollFromResponseToken = React.useCallback(
    (sequence: number, replacementToken?: string) => {
      if (isGenerating) return
      if (responseLogprobs.length === 0) return
      const context = lastGenerationContextRef.current
      if (!context) return
      const rerollPrompt = buildRerollPromptFromRows(responseLogprobs, {
        prefix: context.prefix,
        suffix: context.suffix,
        sequence,
        replacementToken,
        placeholder: PREDICT_PLACEHOLDER
      })
      void handleGenerate(rerollPrompt)
    },
    [handleGenerate, isGenerating, responseLogprobs, lastGenerationContextRef]
  )

  const responseInspectorRowsAll = React.useMemo(
    () =>
      selectResponseInspectorRows(responseLogprobs, {
        query: responseInspectorQuery,
        hideWhitespaceOnly: responseInspectorHideWhitespace,
        sort: responseInspectorSort,
        maxRows: Number.MAX_SAFE_INTEGER
      }),
    [
      responseInspectorHideWhitespace,
      responseInspectorQuery,
      responseInspectorSort,
      responseLogprobs
    ]
  )

  const responseLogprobRows = React.useMemo(
    () => responseInspectorRowsAll.slice(0, MAX_RESPONSE_LOGPROBS),
    [responseInspectorRowsAll]
  )

  const responseLogprobTruncated =
    responseInspectorRowsAll.length > responseLogprobRows.length

  const inlineResponseTokens = React.useMemo(
    () =>
      responseLogprobs.slice(0, MAX_RESPONSE_LOGPROBS).map((entry, sequence) => ({
        sequence,
        displayToken: normalizeInspectorToken(entry.token) || " ",
        topLogprobs: entry.topLogprobs.slice(0, 10).map((alt) => ({
          token: alt.token,
          displayToken: normalizeInspectorToken(alt.token) || " ",
          probability: Math.exp(alt.logprob)
        }))
      })),
    [responseLogprobs]
  )

  const inlineResponseTokensTruncated =
    responseLogprobs.length > inlineResponseTokens.length

  const handleExportResponseInspectorCsv = React.useCallback(() => {
    if (responseInspectorRowsAll.length === 0) return
    const csv = buildResponseInspectorCsv(responseInspectorRowsAll)
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = "writing-response-inspector.csv"
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }, [responseInspectorRowsAll])

  // --- Token inspector derived ---
  const tokenPreviewRows = React.useMemo(
    () => buildTokenPreviewRows(tokenizeResult?.ids ?? [], tokenizeResult?.strings, 200),
    [tokenizeResult]
  )
  const tokenPreviewRawText = React.useMemo(
    () => joinTokenStrings(tokenizeResult?.strings),
    [tokenizeResult?.strings]
  )
  const tokenPreviewTotal = tokenizeResult?.ids.length ?? 0
  const tokenPreviewTruncated = tokenPreviewTotal > tokenPreviewRows.length

  // --- Feature flags ---
  const serverSupportsTokenize = writingCaps?.server?.tokenize === true
  const serverSupportsTokenCount = writingCaps?.server?.token_count === true
  const serverSupportsWordclouds = writingCaps?.server?.wordclouds === true
  const showResponseInspectorPanel =
    settings.logprobs || responseLogprobs.length > 0
  const showTokenInspectorPanel =
    serverSupportsTokenCount || serverSupportsTokenize
  const showWordcloudPanel = serverSupportsWordclouds
  const requestedTokenizerAvailable =
    requestedCaps?.requested?.tokenizer_available === true
  const requestedTokenizerError =
    typeof requestedCaps?.requested?.tokenization_error === "string"
      ? requestedCaps.requested.tokenization_error.trim() || null
      : null
  const tokenizerName =
    [
      tokenCountResult?.meta.tokenizer,
      tokenizeResult?.meta.tokenizer,
      requestedCaps?.requested?.tokenizer
    ].find((value): value is string => typeof value === "string" && value.length > 0) ??
    null
  const tokenInspectorBusy = isCountingTokens || isTokenizingText
  const responseInspectorHasRows = responseInspectorRowsAll.length > 0

  const canCountTokens =
    !settingsDisabled &&
    !tokenInspectorBusy &&
    serverSupportsTokenCount &&
    !tokenInspectorUnavailableReasonFn()

  const canTokenizePreview =
    !settingsDisabled &&
    !tokenInspectorBusy &&
    serverSupportsTokenize &&
    !tokenInspectorUnavailableReasonFn()

  function tokenInspectorUnavailableReasonFn(): string | null {
    if (!selectedModel) {
      return t(
        "option:writingPlayground.tokenInspectorModelMissing",
        "Select a model to use token inspection."
      )
    }
    if (String(selectedModel).trim().toLowerCase() === AUTO_MODEL_ID) {
      return t(
        "option:writingPlayground.tokenInspectorAutoUnavailable",
        "Token inspection requires a concrete model when Auto routing is enabled."
      )
    }
    if (requestedCapsLoading) {
      return t(
        "option:writingPlayground.tokenInspectorChecking",
        "Checking tokenizer support for this model..."
      )
    }
    if (requestedTokenizerError) {
      return requestedTokenizerError
    }
    if (!requestedTokenizerAvailable) {
      return t(
        "option:writingPlayground.tokenInspectorUnavailable",
        "Tokenizer unavailable for this provider/model."
      )
    }
    return null
  }

  const tokenInspectorUnavailableReason = tokenInspectorUnavailableReasonFn()

  const canGenerateWordcloud =
    !settingsDisabled &&
    !isGeneratingWordcloud &&
    serverSupportsWordclouds

  const wordcloudTopWeight = wordcloudWords[0]?.weight ?? 0
  const wordcloudStatusColor =
    wordcloudStatus === "ready"
      ? "green"
      : wordcloudStatus === "failed"
        ? "red"
        : wordcloudStatus
          ? "blue"
          : "default"

  const logprobsUnavailableReason = React.useMemo(() => {
    if (!selectedModel) {
      return t(
        "option:writingPlayground.logprobsModelMissing",
        "Select a model to configure logprobs."
      )
    }
    if (requestedCapsLoading) {
      return t(
        "option:writingPlayground.logprobsChecking",
        "Checking logprobs support for this model..."
      )
    }
    if (requestedLogprobsExplicitlyUnsupported) {
      return t(
        "option:writingPlayground.logprobsUnavailable",
        "Logprobs are not advertised for this provider/model."
      )
    }
    return null
  }, [
    requestedCapsLoading,
    requestedLogprobsExplicitlyUnsupported,
    selectedModel,
    t
  ])

  const topLogprobsHint = React.useMemo(() => {
    if (requestedCapsLoading) {
      return t(
        "option:writingPlayground.topLogprobsChecking",
        "Checking top_logprobs metadata..."
      )
    }
    if (requestedLogprobsSupported && requestedTopLogprobsSupported) {
      return t(
        "option:writingPlayground.topLogprobsSupported",
        "Model metadata includes top_logprobs support."
      )
    }
    if (requestedLogprobsSupported && !requestedTopLogprobsSupported) {
      return t(
        "option:writingPlayground.topLogprobsNotAdvertised",
        "top_logprobs is not advertised for this model. Compatibility mode keeps the field optional."
      )
    }
    return t(
      "option:writingPlayground.topLogprobsFallback",
      "Compatibility mode: if supported by the provider, top_logprobs may still be honored."
    )
  }, [
    requestedCapsLoading,
    requestedLogprobsSupported,
    requestedTopLogprobsSupported,
    t
  ])

  return {
    // panel state
    libraryOpen, setLibraryOpen,
    inspectorOpen, setInspectorOpen,
    activeInspectorTab, setActiveInspectorTab,
    showPromptChunks, setShowPromptChunks,
    generationElapsed,
    // token inspector
    tokenCountResult, setTokenCountResult,
    tokenizeResult, setTokenizeResult,
    tokenInspectorError, setTokenInspectorError,
    isCountingTokens,
    isTokenizingText,
    handleCountTokens,
    handleTokenizePreview,
    clearTokenInspector,
    tokenPreviewRows,
    tokenPreviewRawText,
    tokenPreviewTotal,
    tokenPreviewTruncated,
    tokenInspectorBusy,
    tokenInspectorUnavailableReason,
    tokenizerName,
    // response inspector
    responseLogprobs, setResponseLogprobs,
    responseInspectorQuery, setResponseInspectorQuery,
    responseInspectorSort, setResponseInspectorSort,
    responseInspectorHideWhitespace, setResponseInspectorHideWhitespace,
    generationTokenCount, setGenerationTokenCount,
    generationTokensPerSec, setGenerationTokensPerSec,
    clearResponseInspector,
    handleCopyResponseInspectorJson,
    handleRerollFromResponseToken,
    handleExportResponseInspectorCsv,
    responseInspectorRowsAll,
    responseLogprobRows,
    responseLogprobTruncated,
    inlineResponseTokens,
    inlineResponseTokensTruncated,
    responseInspectorHasRows,
    // wordcloud
    wordcloudId,
    wordcloudStatus,
    wordcloudWords,
    wordcloudMeta,
    wordcloudError,
    isGeneratingWordcloud,
    wordcloudMaxWords, setWordcloudMaxWords,
    wordcloudMinWordLength, setWordcloudMinWordLength,
    wordcloudKeepNumbers, setWordcloudKeepNumbers,
    wordcloudStopwordsInput, setWordcloudStopwordsInput,
    handleGenerateWordcloud,
    clearWordcloud,
    wordcloudTopWeight,
    wordcloudStatusColor,
    canGenerateWordcloud,
    // feature flags
    serverSupportsTokenize,
    serverSupportsTokenCount,
    serverSupportsWordclouds,
    showResponseInspectorPanel,
    showTokenInspectorPanel,
    showWordcloudPanel,
    canCountTokens,
    canTokenizePreview,
    logprobsUnavailableReason,
    topLogprobsHint
  }
}
