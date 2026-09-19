import { hasLowMeasuredRelevance } from "../sourceListUtils"
import React, { useEffect, useMemo, useRef, useState } from "react"
import { Loader2 } from "lucide-react"
import { cn } from "@/libs/utils"
import type { QueryStage } from "../types"
import { useKnowledgeQA } from "../KnowledgeQAProvider"
import { ConversationThread } from "../ConversationThread"
import { AnswerPanel } from "../AnswerPanel"
import { FollowUpInput } from "../FollowUpInput"

type AnswerWorkspaceProps = {
  queryStage: QueryStage
  className?: string
}

const STAGE_COPY: Record<QueryStage, string> = {
  idle: "Ready to search",
  searching: "Searching selected sources",
  ranking: "Ranking best evidence",
  generating: "Generating answer",
  verifying: "Checking source citations",
  complete: "Answer complete",
  cancelled: "Search cancelled",
  error: "Search needs attention",
}

const LIVE_STAGE_COPY: Partial<Record<QueryStage, string>> = {
  searching: "Searching your selected sources.",
  ranking: "Ranking retrieved sources.",
  generating: "Generating answer.",
  verifying: "Checking source citations.",
}

type TurnPreview = {
  turnNumber: number
  question: string
  answer: string | null
}

function truncatePreview(value: string, maxLength = 140): string {
  const normalized = value.trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength).trimEnd()}...`
}

export function AnswerWorkspace({ queryStage, className }: AnswerWorkspaceProps) {
  const {
    results = [],
    error = null,
    queryWarning = null,
    messages = [],
    citations = [],
    settings,
  } = useKnowledgeQA()
  const isActiveStage =
    queryStage !== "idle" && queryStage !== "complete" && queryStage !== "error" && queryStage !== "cancelled"
  const [politeAnnouncement, setPoliteAnnouncement] = useState("")
  const [assertiveAnnouncement, setAssertiveAnnouncement] = useState("")
  const previousStageRef = useRef<QueryStage | null>(null)
  const turnPreviews = useMemo<TurnPreview[]>(
    () => {
      const previews: TurnPreview[] = []
      let turnNumber = 0

      for (let index = 0; index < messages.length; index += 1) {
        const message = messages[index]
        if (message?.role !== "user") continue

        turnNumber += 1
        const rawQuestion =
          typeof (message as { content?: unknown })?.content === "string"
            ? String((message as { content?: unknown }).content)
            : ""
        const question =
          rawQuestion.trim().length > 0
            ? truncatePreview(rawQuestion)
            : `Question ${turnNumber}`

        let answer: string | null = null
        for (let candidateIndex = index + 1; candidateIndex < messages.length; candidateIndex += 1) {
          const candidate = messages[candidateIndex]
          if (candidate?.role === "user") break
          if (candidate?.role !== "assistant") continue
          const rawAnswer =
            typeof (candidate as { content?: unknown })?.content === "string"
              ? String((candidate as { content?: unknown }).content)
              : ""
          answer = rawAnswer.trim().length > 0 ? truncatePreview(rawAnswer) : null
          break
        }

        previews.push({
          turnNumber,
          question,
          answer,
        })
      }

      return previews
    },
    [messages]
  )
  const displayedTurnCount = turnPreviews.length
  const priorTurnPreviews = useMemo(
    () => (turnPreviews.length > 1 ? turnPreviews.slice(0, -1).filter((turn) => turn.answer) : []),
    [turnPreviews]
  )
  const hasThreadContext = priorTurnPreviews.length > 0
  const contextSummary = useMemo(() => {
    if (priorTurnPreviews.length === 0) {
      return "The next question starts from the current answer context."
    }
    if (priorTurnPreviews.length === 1) {
      return "Using context from turn 1."
    }
    return `Using context from turns 1-${priorTurnPreviews.length}.`
  }, [priorTurnPreviews.length])

  const isLowQualityResult = useMemo(() => {
    if (queryStage !== "complete") return false
    if (results.length === 0) return false
    const threshold = settings?.strip_min_relevance ?? 0.3
    const allLowRelevance = hasLowMeasuredRelevance(results, threshold)
    const noCitations = (citations?.length ?? 0) === 0
    return allLowRelevance || (noCitations && results.length > 0)
  }, [queryStage, results, citations, settings?.strip_min_relevance])

  useEffect(() => {
    if (queryStage === previousStageRef.current) return
    previousStageRef.current = queryStage

    if (queryStage === "complete") {
      const count = results.length
      setPoliteAnnouncement(
        queryWarning || `Search complete. ${count} source${count === 1 ? "" : "s"} found.`
      )
      return
    }
    if (queryStage === "cancelled") {
      setPoliteAnnouncement("Search cancelled. You can ask again when ready.")
      return
    }
    if (queryStage === "error") {
      setPoliteAnnouncement("")
      return
    }
    const stageMessage = LIVE_STAGE_COPY[queryStage]
    if (stageMessage) {
      setPoliteAnnouncement(stageMessage)
    }
  }, [queryStage, results.length, queryWarning])

  useEffect(() => {
    setAssertiveAnnouncement(error ? `Search error. ${error}` : "")
  }, [error])

  return (
    <div className={cn("space-y-6", className)}>
      <div
        className={queryStage === "cancelled" ? "rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm text-text-muted" : "sr-only"}
        role={queryStage === "cancelled" ? "status" : undefined}
        aria-live="polite"
        aria-atomic="true"
      >
        {politeAnnouncement}
      </div>
      <div className="sr-only" aria-live="assertive" aria-atomic="true">
        {assertiveAnnouncement}
      </div>

      {isActiveStage ? (
        <div className="rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm text-text-muted">
          <span className="inline-flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            {STAGE_COPY[queryStage]}
          </span>
        </div>
      ) : null}

      {hasThreadContext ? (
        <div className="rounded-lg border border-border bg-surface/70 px-3 py-2 text-xs text-text-muted">
          <p className="font-medium text-text">
            Conversation • {displayedTurnCount} turn
            {displayedTurnCount === 1 ? "" : "s"}
          </p>
          <p className="mt-1">{contextSummary}</p>
          {priorTurnPreviews.length > 0 ? (
            <article className="mt-2 rounded-md border border-border bg-muted/20 px-2 py-2">
              <p className="text-[11px] font-medium text-text">Previous turn</p>
              <p className="mt-1 text-[11px] leading-relaxed">
                <span className="font-medium text-text">Q:</span>{" "}
                {priorTurnPreviews[priorTurnPreviews.length - 1]?.question}
              </p>
              <p className="mt-1 text-[11px] leading-relaxed">
                <span className="font-medium text-text">A:</span>{" "}
                {priorTurnPreviews[priorTurnPreviews.length - 1]?.answer ||
                  "No answer recorded."}
              </p>
              {priorTurnPreviews.length > 1 ? (
                <p className="mt-2 text-[11px] text-text-muted">
                  {priorTurnPreviews.length - 1} more prior turn
                  {priorTurnPreviews.length === 2 ? "" : "s"} are available in thread history.
                </p>
              ) : null}
            </article>
          ) : null}
        </div>
      ) : null}

      <ConversationThread />
      <AnswerPanel />
      <FollowUpInput mode={isLowQualityResult ? "recovery" : "default"} />
    </div>
  )
}
