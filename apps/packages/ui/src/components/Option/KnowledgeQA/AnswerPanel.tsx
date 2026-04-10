/**
 * AnswerPanel - Displays generated answer with inline citations
 */

import React, { type ReactNode, useEffect, useMemo, useRef, useState } from "react"
import { Sparkles, AlertCircle, Loader2, ThumbsUp, ThumbsDown } from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import { getFeedbackSessionId, submitExplicitFeedback } from "@/services/feedback"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useNavigate } from "react-router-dom"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  buildKnowledgeQaWorkspacePrefill,
  queueWorkspacePlaygroundPrefill,
} from "@/utils/workspace-playground-prefill"
import { trackKnowledgeQaSearchMetric } from "@/utils/knowledge-qa-search-metrics"
import { remarkCitationLinks } from "./answerMarkdown"
import { LowQualityRecoveryBanner } from "./panels/LowQualityRecoveryBanner"

type AnswerPanelProps = {
  className?: string
}

const LONG_ANSWER_WORD_THRESHOLD = 1000

type ErrorPresentation = {
  title: string
  guidance: string
}

type GroundingCoverage = {
  citedSentences: number
  totalSentences: number
  percent: number
}

type TrustDescriptor = {
  label: "Strong" | "Partial" | "Weak"
  className: string
}

const classifyErrorMessage = (rawMessage: string): ErrorPresentation => {
  const message = rawMessage.trim()
  if (/timed out|timeout/i.test(message)) {
    return {
      title: "Search timed out",
      guidance: "Try the Fast preset or reduce the scope of your query.",
    }
  }
  if (/cannot reach server|network|offline|connection/i.test(message)) {
    return {
      title: "Cannot reach server",
      guidance: "Check your connection and server status, then try again.",
    }
  }
  if (/no relevant documents|no results/i.test(message)) {
    return {
      title: "No relevant documents found",
      guidance: "Try broader keywords or verify your sources are indexed.",
    }
  }
  return {
    title: "Search failed",
    guidance: "Try adjusting your query or settings and run the search again.",
  }
}

function extractTextFromNode(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node)
  }
  if (!node) {
    return ""
  }
  if (Array.isArray(node)) {
    return node.map((entry) => extractTextFromNode(entry)).join("")
  }
  if (React.isValidElement<{ children?: ReactNode }>(node)) {
    return extractTextFromNode(node.props.children)
  }
  return ""
}

function computeGroundingCoverage(answer: string | null): GroundingCoverage | null {
  if (!answer || answer.trim().length === 0) return null

  const sentences = answer
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => /[A-Za-z0-9]/.test(sentence))

  if (sentences.length === 0) {
    return null
  }

  const citedSentences = sentences.filter((sentence) => /\[\d+\]/.test(sentence)).length
  const totalSentences = sentences.length
  const percent = Math.round((citedSentences / totalSentences) * 100)

  return {
    citedSentences,
    totalSentences,
    percent,
  }
}

function describeFaithfulness(score: number | null): TrustDescriptor | null {
  if (score == null || Number.isNaN(score)) return null
  if (score >= 0.85) {
    return {
      label: "Strong",
      className: "border-success/40 bg-success/15 text-success",
    }
  }
  if (score >= 0.6) {
    return {
      label: "Partial",
      className: "border-warn/40 bg-warn/15 text-warn",
    }
  }
  return {
    label: "Weak",
    className: "border-danger/40 bg-danger/15 text-danger",
  }
}

export function AnswerPanel({ className }: AnswerPanelProps) {
  const {
    answer,
    citations,
    isSearching,
    error,
    scrollToSource,
    focusedSourceIndex = null,
    results,
    searchDetails,
    query = "",
    currentThreadId = null,
    messages = [],
    setSettingsPanelOpen,
    updateSetting,
    preset,
    settings,
    rerunWithTokenLimit,
    expertMode = false,
  } = useKnowledgeQA()
  const [isExpanded, setIsExpanded] = useState(false)
  const [answerFeedback, setAnswerFeedback] = useState<"up" | "down" | null>(null)
  const [answerFeedbackSubmitting, setAnswerFeedbackSubmitting] = useState(false)
  const [answerFeedbackError, setAnswerFeedbackError] = useState<string | null>(null)
  const [pendingFeedbackThumb, setPendingFeedbackThumb] = useState<"up" | "down" | null>(
    null
  )
  const [workspaceHandoffPending, setWorkspaceHandoffPending] = useState(false)
  const [answerLengthAction, setAnswerLengthAction] = useState<"shorter" | "longer" | null>(
    null
  )
  const [recoveryDismissed, setRecoveryDismissed] = useState(false)
  const [copiedAnswer, setCopiedAnswer] = useState(false)
  const [loadingElapsedSeconds, setLoadingElapsedSeconds] = useState(0)
  const copiedAnswerTimeoutRef = useRef<number | null>(null)
  const activeAnswerSessionKeyRef = useRef("")
  const messageApi = useAntdMessage()
  const navigate = useNavigate()

  // Get cited indices for highlighting
  const citedIndices = useMemo(() => citations.map((c) => c.index), [citations])
  const latestAssistantMessageId = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "assistant") {
        return messages[index].id
      }
    }
    return null
  }, [messages])
  const normalizedAnswer = useMemo(() => {
    if (typeof answer !== "string") return null
    return answer.trim().length > 0 ? answer : null
  }, [answer])
  const answerSessionKey = useMemo(
    () =>
      `${currentThreadId ?? "no-thread"}::${latestAssistantMessageId ?? "no-assistant"}::${
        normalizedAnswer ?? ""
      }`,
    [currentThreadId, latestAssistantMessageId, normalizedAnswer]
  )
  const answerWordCount = useMemo(() => {
    if (!normalizedAnswer) return 0
    return normalizedAnswer.trim().split(/\s+/).filter(Boolean).length
  }, [normalizedAnswer])
  const isLongAnswer = answerWordCount > LONG_ANSWER_WORD_THRESHOLD
  const groundingCoverage = useMemo(
    () => computeGroundingCoverage(normalizedAnswer),
    [normalizedAnswer]
  )
  const trustScore = searchDetails?.faithfulnessScore ?? searchDetails?.verificationRate ?? null
  const trustScoreLabel = searchDetails?.faithfulnessScore != null
    ? "How well claims are backed by sources"
    : "How well claims are backed by sources"
  const faithfulnessDescriptor = useMemo(
    () => describeFaithfulness(trustScore),
    [trustScore]
  )
  const lowConfidenceRecovery = useMemo(() => {
    if (!normalizedAnswer || results.length === 0) return null

    const threshold = settings?.strip_min_relevance ?? 0.3
    const hasScoredResults = results.some(
      (result: { score?: number }) => typeof result.score === "number"
    )
    const allLowRelevance =
      hasScoredResults &&
      results.every(
        (result: { score?: number }) =>
          typeof result.score === "number" && result.score < threshold
      )
    const uncitedAnswer = citations.length === 0 || groundingCoverage?.percent === 0
    const weakVerification = faithfulnessDescriptor?.label === "Weak"

    if (uncitedAnswer && allLowRelevance) {
      return {
        title: "Low answer confidence",
        description:
          "This answer does not cite the retrieved sources, and the current matches are weak. Refine the search before relying on it.",
      }
    }

    if (uncitedAnswer) {
      return {
        title: "Low answer confidence",
        description:
          "This answer does not cite the retrieved sources. Refine the search or broaden the evidence before trusting it.",
      }
    }

    if (weakVerification) {
      return {
        title: "Low answer confidence",
        description:
          "Server-side claim checks found weak source support for this answer. Tighten the query or inspect your sources before relying on it.",
      }
    }

    if (allLowRelevance) {
      return {
        title: "Low answer confidence",
        description:
          "The retrieved sources scored below your current relevance threshold. Narrow the question or adjust your sources to improve grounding.",
      }
    }

    return null
  }, [
    citations.length,
    faithfulnessDescriptor?.label,
    groundingCoverage?.percent,
    normalizedAnswer,
    results,
    settings?.strip_min_relevance,
  ])
  const showLowConfidenceRecovery = lowConfidenceRecovery !== null && !recoveryDismissed

  useEffect(() => {
    activeAnswerSessionKeyRef.current = answerSessionKey
  }, [answerSessionKey])

  useEffect(() => {
    setIsExpanded(false)
    setAnswerLengthAction(null)
    setAnswerFeedback(null)
    setAnswerFeedbackSubmitting(false)
    setAnswerFeedbackError(null)
    setPendingFeedbackThumb(null)
    setWorkspaceHandoffPending(false)
    setRecoveryDismissed(false)
    if (copiedAnswerTimeoutRef.current != null) {
      window.clearTimeout(copiedAnswerTimeoutRef.current)
      copiedAnswerTimeoutRef.current = null
    }
    setCopiedAnswer(false)
  }, [answerSessionKey])

  useEffect(() => {
    if (!isSearching) {
      setLoadingElapsedSeconds(0)
      return
    }

    setLoadingElapsedSeconds(0)
    const intervalId = window.setInterval(() => {
      setLoadingElapsedSeconds((prev) => prev + 1)
    }, 1000)

    return () => window.clearInterval(intervalId)
  }, [isSearching])

  useEffect(() => {
    return () => {
      if (copiedAnswerTimeoutRef.current != null) {
        window.clearTimeout(copiedAnswerTimeoutRef.current)
        copiedAnswerTimeoutRef.current = null
      }
    }
  }, [])

  const handleCitationClick = (index: number) => {
    scrollToSource(index - 1) // Convert from 1-based to 0-based
  }

  const handleSubmitAnswerFeedback = async (thumb: "up" | "down") => {
    const requestSessionKey = answerSessionKey
    setAnswerFeedbackSubmitting(true)
    setPendingFeedbackThumb(thumb)
    setAnswerFeedbackError(null)
    try {
      await submitExplicitFeedback({
        conversation_id: currentThreadId || undefined,
        message_id: latestAssistantMessageId || undefined,
        query: query?.trim() || undefined,
        feedback_type: "helpful",
        helpful: thumb === "up",
        feedback_id: searchDetails?.feedbackId || undefined,
        session_id: getFeedbackSessionId(),
      })
      if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
        return
      }
      void trackKnowledgeQaSearchMetric({
        type: "answer_feedback_submit",
        helpful: thumb === "up",
      })
      setAnswerFeedback(thumb)
      messageApi.open({
        type: "success",
          content: "Feedback submitted.",
          duration: 2,
      })
    } catch (submissionError) {
      if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
        return
      }
      const detail =
        submissionError instanceof Error && submissionError.message
          ? submissionError.message
          : "Unable to submit feedback right now."
      setAnswerFeedbackError(detail)
      messageApi.open({
        type: "error",
        content: "Feedback could not be sent. You can retry.",
        duration: 3,
      })
    } finally {
      if (activeAnswerSessionKeyRef.current === requestSessionKey) {
        setAnswerFeedbackSubmitting(false)
      }
    }
  }

  const citationRenderCounts = new Map<number, number>()
  const getNextCitationOccurrence = (citationNum: number): number => {
    const nextOccurrence = (citationRenderCounts.get(citationNum) || 0) + 1
    citationRenderCounts.set(citationNum, nextOccurrence)
    return nextOccurrence
  }

  const renderCitationButton = (citationNum: number, occurrence: number) => {
    const isCited = citedIndices.includes(citationNum)
    const isFocusedCitation = focusedSourceIndex === citationNum - 1
    return (
      <button
        type="button"
        onClick={() => handleCitationClick(citationNum)}
        aria-label={`Jump to source ${citationNum}`}
        aria-current={isFocusedCitation ? "true" : undefined}
        data-knowledge-citation-index={citationNum}
        data-knowledge-citation-occurrence={occurrence}
        className={cn(
          "inline-flex items-center justify-center",
          "min-w-8 h-8 px-2 mx-0.5 sm:min-w-[1.5rem] sm:h-5 sm:px-1.5",
          "text-sm sm:text-xs font-medium rounded",
          "transition-colors duration-200",
          isCited
            ? "bg-primary text-white dark:text-slate-900 hover:brightness-105"
            : "bg-surface text-text-subtle border border-border hover:bg-hover hover:text-text",
          isFocusedCitation && "ring-2 ring-primary/40 ring-offset-1 ring-offset-surface"
        )}
        title={`Jump to source ${citationNum}`}
      >
        {citationNum}
      </button>
    )
  }

  const handleOpenInWorkspace = async () => {
    if (workspaceHandoffPending) return

    const requestSessionKey = answerSessionKey
    setWorkspaceHandoffPending(true)
    try {
      const payload = buildKnowledgeQaWorkspacePrefill({
        threadId: currentThreadId,
        query,
        answer: normalizedAnswer,
        citations: citations.map((citation) => citation.index),
        results,
      })
      await queueWorkspacePlaygroundPrefill(payload)
      if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
        return
      }
      void trackKnowledgeQaSearchMetric({
        type: "workspace_handoff",
        source_count: results.length,
      })
      navigate("/workspace-playground")
    } catch (error) {
      if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
        return
      }
      console.error("Failed to open workspace with Knowledge QA context:", error)
      messageApi.open({
        type: "error",
        content: "Unable to open Workspace right now.",
        duration: 3,
      })
    } finally {
      if (activeAnswerSessionKeyRef.current === requestSessionKey) {
        setWorkspaceHandoffPending(false)
      }
    }
  }

  const handleCopyAnswer = async () => {
    if (!normalizedAnswer) return
    const requestSessionKey = answerSessionKey
    try {
      await navigator.clipboard.writeText(normalizedAnswer)
      if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
        return
      }
      setCopiedAnswer(true)
      if (copiedAnswerTimeoutRef.current != null) {
        window.clearTimeout(copiedAnswerTimeoutRef.current)
      }
      copiedAnswerTimeoutRef.current = window.setTimeout(() => {
        if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
          return
        }
        copiedAnswerTimeoutRef.current = null
        setCopiedAnswer(false)
      }, 2000)
      messageApi.open({
        type: "success",
        content: "Answer copied.",
        duration: 2,
      })
    } catch {
      if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
        return
      }
      messageApi.open({
        type: "error",
        content: "Unable to copy answer.",
        duration: 3,
      })
    }
  }

  const handleAdjustAnswerLength = async (action: "shorter" | "longer") => {
    if (!query?.trim() || isSearching) return
    const requestSessionKey = answerSessionKey
    const baseTokens =
      typeof settings?.max_generation_tokens === "number"
        ? settings.max_generation_tokens
        : 800
    const nextLimit =
      action === "shorter"
        ? Math.max(200, Math.round(baseTokens * 0.6))
        : Math.min(4000, Math.round(baseTokens * 1.5))

    setAnswerLengthAction(action)
    try {
      await rerunWithTokenLimit(nextLimit)
    } finally {
      if (activeAnswerSessionKeyRef.current === requestSessionKey) {
        setAnswerLengthAction(null)
      }
    }
  }

  const loadingStageLabel = useMemo(() => {
    if (loadingElapsedSeconds < 5) return "Searching documents..."
    if (loadingElapsedSeconds < 10) return "Reranking results..."
    if (loadingElapsedSeconds < 20) return "Generating answer..."
    return "Verifying citations..."
  }, [loadingElapsedSeconds])

  const presetLatencyHint = useMemo(() => {
    if (preset === "fast") {
      return "Fast preset usually completes in a few seconds."
    }
    if (preset === "balanced") {
      return "Balanced preset typically completes within about 10 seconds."
    }
    if (preset === "thorough") {
      return "Thorough preset may take up to 30 seconds."
    }
    return "Custom preset timing varies with your settings."
  }, [preset])

  // Loading state
  if (isSearching && !normalizedAnswer) {
    return (
      <div className={cn("p-6 rounded-xl bg-muted/30 border border-border", className)}>
        <div className="flex items-center gap-3">
          <Loader2 className="w-5 h-5 animate-spin text-primary" />
          <div>
            <p className="font-medium">
              {loadingStageLabel}{" "}
              {loadingElapsedSeconds > 0 && (
                <span className="text-text-muted">({loadingElapsedSeconds}s)</span>
              )}
            </p>
            <p className="text-sm text-text-muted">
              {presetLatencyHint}
            </p>
          </div>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    const errorPresentation = classifyErrorMessage(error)
    return (
      <div className={cn("p-6 rounded-xl bg-danger/10 border border-danger/20", className)}>
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-danger mt-0.5" />
          <div>
            <p className="font-medium text-danger">{errorPresentation.title}</p>
            <p className="text-sm text-text-muted mt-1">{error}</p>
            <p className="text-sm text-text-muted mt-1">{errorPresentation.guidance}</p>
          </div>
        </div>
      </div>
    )
  }

  // No answer yet
  if (!normalizedAnswer) {
    // Show empty state only if we have no results either
    if (results.length === 0) {
      return null
    }

    // Results but no generated answer
    return (
      <div className={cn("p-6 rounded-xl bg-muted/30 border border-border", className)}>
        <div className="flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-text-muted mt-0.5" />
          <div>
            <p className="text-text-muted">
              Found {results.length} relevant source{results.length !== 1 ? "s" : ""}.
              Enable answer generation in settings to get a synthesized response.
            </p>
            <button
              type="button"
              onClick={() => setSettingsPanelOpen(true)}
              className="mt-2 inline-flex items-center rounded-md border border-primary/30 bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/15 transition-colors"
            >
              Enable in Settings
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Answer with citations
  return (
    <div className={cn("rounded-xl bg-gradient-to-br from-primary/5 to-primary/10 border border-primary/20", className)}>
      {/* Header */}
      <div className="border-b border-primary/10 px-6 py-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="font-medium text-sm">AI Answer</span>
          {isSearching && (
            <span className="text-xs text-text-muted">Streaming...</span>
          )}
          {searchDetails?.webFallbackTriggered && (
            <span className="inline-flex items-center rounded-md border border-primary/30 bg-primary/10 px-2 py-0.5 text-xs text-primary">
              Includes web sources
              {searchDetails.webFallbackEngine
                ? ` (${searchDetails.webFallbackEngine})`
                : ""}
            </span>
          )}
          {citations.length > 0 && (
            <span className="text-xs text-text-muted">
              {citations.length} citation{citations.length !== 1 ? "s" : ""}
            </span>
          )}
          {/* Grounding coverage badge: expert mode only */}
          {expertMode && groundingCoverage && !showLowConfidenceRecovery ? (
            groundingCoverage.percent > 0 ? (
              <span
                className="inline-flex items-center rounded-md border border-border bg-surface px-2 py-0.5 text-xs text-text-muted"
                title={`${groundingCoverage.citedSentences}/${groundingCoverage.totalSentences} answer sentences include citations.`}
              >
                {groundingCoverage.percent}% of answer cites sources
              </span>
            ) : null
          ) : null}
          {/* Faithfulness badge: always in expert mode, non-expert only when below Strong */}
          {faithfulnessDescriptor &&
          (expertMode || faithfulnessDescriptor.label !== "Strong") ? (
            <span
              className={cn(
                "inline-flex items-center rounded-md border px-2 py-0.5 text-xs",
                faithfulnessDescriptor.className
              )}
              title={`${trustScoreLabel} score ${Math.round((trustScore || 0) * 100)}% based on server-side claim verification.`}
            >
              Source support: {faithfulnessDescriptor.label}
            </span>
          ) : null}
          {searchDetails?.verificationReportAvailable ? (
            <span
              className="inline-flex items-center rounded-md border border-border px-2 py-0.5 text-xs text-text-muted"
              title="Structured claim check report is available for this answer."
            >
              Claim check
              {searchDetails.verificationTotalClaims != null
                ? ` (${searchDetails.verificationTotalClaims} claims)`
                : ""}
            </span>
          ) : null}
        </div>
        <div
          data-testid="knowledge-answer-header-actions"
          className="mt-3 flex flex-wrap items-center justify-between gap-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void handleAdjustAnswerLength("shorter")
              }}
              disabled={isSearching || answerLengthAction !== null}
              className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-text-subtle hover:bg-hover hover:text-text disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {answerLengthAction === "shorter" ? "Summarizing..." : "Summarize"}
            </button>
            <button
              type="button"
              onClick={() => {
                void handleAdjustAnswerLength("longer")
              }}
              disabled={isSearching || answerLengthAction !== null}
              className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-text-subtle hover:bg-hover hover:text-text disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {answerLengthAction === "longer" ? "Expanding..." : "Show more"}
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void handleCopyAnswer()
              }}
              className="rounded-md border px-2 py-1 text-xs transition-colors border-border bg-surface text-text-subtle hover:bg-hover hover:text-text"
            >
              {copiedAnswer ? "Copied" : "Copy answer"}
            </button>
            <button
              type="button"
              onClick={() => {
                void handleOpenInWorkspace()
              }}
              disabled={workspaceHandoffPending}
              className={cn(
                "rounded-md border px-2 py-1 text-xs transition-colors",
                "border-border bg-surface text-text-subtle hover:bg-hover hover:text-text",
                workspaceHandoffPending && "opacity-60 cursor-not-allowed"
              )}
            >
              {workspaceHandoffPending ? "Opening..." : "Continue in editor"}
            </button>
          </div>
        </div>
      </div>
      {showLowConfidenceRecovery ? (
        <div className="border-b border-primary/10 px-6 py-4">
          <LowQualityRecoveryBanner
            title={lowConfidenceRecovery?.title}
            description={lowConfidenceRecovery?.description}
            refineLabel="Refine search"
            enableWebLabel="Include web sources"
            selectSourcesLabel="Adjust sources"
            onRefine={() => {
              const input = document.getElementById("knowledge-search-input")
              input?.focus()
            }}
            onEnableWeb={() => {
              updateSetting("enable_web_fallback", true)
            }}
            onSelectSources={() => {
              setSettingsPanelOpen(true)
            }}
            onDismiss={() => setRecoveryDismissed(true)}
          />
        </div>
      ) : null}

      {/* Answer content */}
      <div className="p-6">
        <div className="mx-auto w-full max-w-3xl xl:max-w-4xl">
          {citations.length > 0 ? (
            <p id="knowledge-answer-citation-guidance" className="sr-only">
              This answer includes inline citation buttons. Press Tab to move between
              citation controls and source links.
            </p>
          ) : null}
          <div
            id="knowledge-answer-content"
            data-testid="knowledge-answer-content"
            aria-describedby={
              citations.length > 0 ? "knowledge-answer-citation-guidance" : undefined
            }
            className={cn(
              "prose prose-sm dark:prose-invert max-w-none",
              isLongAnswer && !isExpanded && "relative max-h-[28rem] overflow-hidden"
            )}
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkCitationLinks]}
              components={{
                a({ href, children, ...props }) {
                  const citationMatch =
                    typeof href === "string"
                      ? href.match(/^citation:\/\/(\d+)$/)
                      : null
                  const childText =
                    typeof children === "string"
                      ? children
                      : Array.isArray(children) && children.length === 1
                        ? typeof children[0] === "string"
                          ? children[0]
                          : null
                        : null
                  const childCitationMatch =
                    typeof childText === "string"
                      ? childText.match(/^\[(\d+)\]$/)
                      : null
                  const citationNum = citationMatch
                    ? Number.parseInt(citationMatch[1], 10)
                    : childCitationMatch
                      ? Number.parseInt(childCitationMatch[1], 10)
                      : NaN
                  if (Number.isFinite(citationNum)) {
                    return renderCitationButton(
                      citationNum,
                      getNextCitationOccurrence(citationNum)
                    )
                  }
                  return (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline"
                      {...props}
                    >
                      {children}
                    </a>
                  )
                },
                code({ className: codeClassName, children, ...props }) {
                  const isInline = !String(codeClassName || "").includes("language-")
                  if (isInline) {
                    return (
                      <code
                        className="rounded bg-muted px-1 py-0.5 text-[0.9em]"
                        {...props}
                      >
                        {children}
                      </code>
                    )
                  }
                  return (
                    <code className={codeClassName} {...props}>
                      {children}
                    </code>
                  )
                },
                pre({ children, ...props }) {
                  return (
                    <pre
                      className="overflow-x-auto rounded-md border border-border bg-surface2/70 p-3"
                      {...props}
                    >
                      {children}
                    </pre>
                  )
                },
                table({ children, ...props }) {
                  return (
                    <table className="block overflow-x-auto text-sm" {...props}>
                      {children}
                    </table>
                  )
                },
                p({ children, ...props }) {
                  const paragraphText = extractTextFromNode(children)
                  const paragraphHasCitation = /\[\d+\]/.test(paragraphText)
                  const shouldHighlightUncited =
                    Boolean(citations.length) &&
                    paragraphText.trim().length > 0 &&
                    !paragraphHasCitation
                  return (
                    <p
                      className={cn(
                        "leading-relaxed whitespace-pre-wrap",
                        shouldHighlightUncited &&
                          "rounded-md bg-amber-500/10 px-2 py-1"
                      )}
                      {...props}
                    >
                      {children}
                    </p>
                  )
                },
              }}
            >
              {normalizedAnswer}
            </ReactMarkdown>
            {isLongAnswer && !isExpanded && (
              <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-primary/10 to-transparent" />
            )}
          </div>
        </div>
        {isLongAnswer && (
          <button
            type="button"
            onClick={() => setIsExpanded((prev) => !prev)}
            className="mt-3 text-sm font-medium text-primary hover:text-primaryStrong transition-colors"
          >
            {isExpanded ? "Show less" : "Show full answer"}
          </button>
        )}
      </div>

      {/* Citation summary + feedback */}
      <div className="border-t border-primary/10 bg-surface/35 px-6 py-3">
        {citations.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-text-muted">Sources:</span>
            {citations.map((citation) => {
              const isFocusedCitation = focusedSourceIndex === citation.index - 1
              return (
                <button
                  type="button"
                  key={citation.index}
                  onClick={() => handleCitationClick(citation.index)}
                  aria-label={`Jump to source ${citation.index}`}
                  aria-current={isFocusedCitation ? "true" : undefined}
                  data-knowledge-citation-index={citation.index}
                  className={cn(
                    "inline-flex h-8 min-w-8 items-center justify-center rounded border border-border bg-surface px-2 text-sm text-text-muted transition-colors hover:bg-surface2 hover:text-text sm:h-6 sm:min-w-[1.5rem] sm:px-1.5 sm:text-xs",
                    isFocusedCitation &&
                      "border-primary/60 bg-primary/10 text-primary ring-2 ring-primary/30"
                  )}
                >
                  [{citation.index}]
                </button>
              )
            })}
          </div>
        )}
        {searchDetails?.tokensUsed != null || searchDetails?.estimatedCostUsd != null ? (
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-text-muted">
            {searchDetails?.tokensUsed != null && (
              <span>
                Used ~{Math.round(searchDetails.tokensUsed).toLocaleString()} tokens
              </span>
            )}
            {searchDetails?.estimatedCostUsd != null && (
              <span>
                Estimated cost ${searchDetails.estimatedCostUsd.toFixed(4)}
              </span>
            )}
          </div>
        ) : null}
        <div
          data-testid="knowledge-answer-feedback"
          className={cn(
            "flex flex-wrap items-center gap-2 text-xs",
            (citations.length > 0 ||
              searchDetails?.tokensUsed != null ||
              searchDetails?.estimatedCostUsd != null) &&
              "mt-3"
          )}
        >
          <span className="text-text-muted md:whitespace-nowrap">Was this answer helpful?</span>
          <button
            type="button"
            onClick={() => {
              if (answerFeedback === "up") return
              void handleSubmitAnswerFeedback("up")
            }}
            disabled={answerFeedbackSubmitting || answerFeedback === "up"}
            aria-pressed={answerFeedback === "up"}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-2 py-1 transition-colors",
              answerFeedback === "up"
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-surface text-text-subtle hover:text-text hover:bg-hover",
              (answerFeedbackSubmitting || answerFeedback === "up") && "opacity-60 cursor-not-allowed"
            )}
          >
            <ThumbsUp className="w-3.5 h-3.5" />
            Helpful
          </button>
          <button
            type="button"
            onClick={() => {
              if (answerFeedback === "down") return
              void handleSubmitAnswerFeedback("down")
            }}
            disabled={answerFeedbackSubmitting || answerFeedback === "down"}
            aria-pressed={answerFeedback === "down"}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-2 py-1 transition-colors",
              answerFeedback === "down"
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-surface text-text-subtle hover:text-text hover:bg-hover",
              (answerFeedbackSubmitting || answerFeedback === "down") && "opacity-60 cursor-not-allowed"
            )}
          >
            <ThumbsDown className="w-3.5 h-3.5" />
            Needs work
          </button>
          {answerFeedbackError && pendingFeedbackThumb && (
            <button
              type="button"
              onClick={() => {
                void handleSubmitAnswerFeedback(pendingFeedbackThumb)
              }}
              className="text-primary underline hover:opacity-80"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
