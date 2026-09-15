/**
 * ConversationThread - Inline history of prior Q&A turns
 */

import React, { useCallback, useEffect, useMemo, useState } from "react"
import { GitBranch, MessageSquare, Sparkles } from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import type { KnowledgeQAMessage, RagContextData } from "./types"
import { useKnowledgeQaBranching } from "@/hooks/useFeatureFlags"
import { createComparisonDraft, isComparisonReady } from "./comparisonModel"
import { tldwClient } from "@/services/tldw/TldwApiClient"

type ConversationThreadProps = {
  className?: string
}

type ConversationTurn = {
  id: string
  threadId: string
  question: string
  answer: string | null
  sourceCount: number
  citationCount: number
  citationIndices: number[]
  timestamp: string
}

type ThreadOption = {
  id: string
  label: string
}

function truncatePreview(value: string, maxLength = 220): string {
  const normalized = value.trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength).trimEnd()}...`
}

function extractInlineCitationIndices(content: string | null | undefined): number[] {
  if (typeof content !== "string" || content.length === 0) return []
  const matches = content.match(/\[(\d+)\]/g) || []
  const values = Array.from(
    new Set(
      matches
        .map((match) => Number.parseInt(match.replace(/[\[\]]/g, ""), 10))
        .filter((value) => Number.isFinite(value) && value > 0)
    )
  )
  return values.sort((left, right) => left - right)
}

function buildConversationTurns(
  messages: KnowledgeQAMessage[],
  threadId: string
): ConversationTurn[] {
  const turns: ConversationTurn[] = []

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]
    if (message.role !== "user") continue

    const nextMessage = messages
      .slice(index + 1)
      .find((candidate) => candidate.role === "assistant" || candidate.role === "user")
    const assistantMessage = nextMessage?.role === "assistant" ? nextMessage : undefined
    const ragCitations = assistantMessage?.ragContext?.citations
    const explicitRagCitationIndices = Array.isArray(ragCitations)
      ? ragCitations
          .map((citation) =>
            typeof citation?.index === "number" && Number.isFinite(citation.index)
              ? Math.max(1, Math.floor(citation.index))
              : null
          )
          .filter((value): value is number => value != null)
      : []
    const ragCitationIndices =
      explicitRagCitationIndices.length > 0
        ? explicitRagCitationIndices
        : Array.isArray(ragCitations)
          ? Array.from({ length: ragCitations.length }, (_, offset) => offset + 1)
          : []
    const inlineCitationIndices = extractInlineCitationIndices(assistantMessage?.content)
    const citationIndices = Array.from(
      new Set([...ragCitationIndices, ...inlineCitationIndices])
    ).sort((left, right) => left - right)

    turns.push({
      id: message.id,
      threadId,
      question: message.content,
      answer: assistantMessage?.content || null,
      sourceCount: assistantMessage?.ragContext?.retrieved_documents?.length || 0,
      citationCount: citationIndices.length,
      citationIndices,
      timestamp: message.timestamp,
    })
  }

  return turns
}

function formatTurnTimestamp(timestamp: string): string {
  const value = new Date(timestamp)
  if (Number.isNaN(value.getTime())) return ""
  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

function normalizeRemoteMessages(
  payload: unknown,
  threadId: string
): KnowledgeQAMessage[] {
  if (!Array.isArray(payload)) return []

  return payload
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null
      const candidate = entry as Record<string, unknown>
      const idRaw = candidate.id
      const id =
        typeof idRaw === "string" && idRaw.length > 0
          ? idRaw
          : `thread-message-${index + 1}`
      const roleRaw = String(candidate.role ?? "").toLowerCase()
      const role: KnowledgeQAMessage["role"] =
        roleRaw === "assistant" || roleRaw === "user" || roleRaw === "system"
          ? roleRaw
          : "system"
      const timestampRaw = [
        candidate.timestamp,
        candidate.created_at,
        candidate.createdAt,
        candidate.updated_at,
      ].find((value) => typeof value === "string" && value.length > 0)
      const ragContextRaw = candidate.ragContext ?? candidate.rag_context
      return {
        id,
        conversationId: threadId,
        role,
        content:
          typeof candidate.content === "string"
            ? candidate.content
            : String(candidate.content ?? ""),
        timestamp:
          typeof timestampRaw === "string" ? timestampRaw : new Date().toISOString(),
        ragContext:
          ragContextRaw && typeof ragContextRaw === "object"
            ? (ragContextRaw as RagContextData)
            : undefined,
      } as KnowledgeQAMessage
    })
    .filter((message): message is KnowledgeQAMessage => message != null)
}

function getTurnsForThread(
  threadId: string | null,
  currentThreadId: string | null,
  currentTurns: ConversationTurn[],
  loadedTurnsByThread: Record<string, ConversationTurn[]>
): ConversationTurn[] {
  if (!threadId) return []
  if (currentThreadId && threadId === currentThreadId) {
    return currentTurns
  }
  return loadedTurnsByThread[threadId] || []
}

export function ConversationThread({ className }: ConversationThreadProps) {
  const {
    messages,
    setQuery,
    branchFromTurn,
    searchHistory,
    currentThreadId,
  } = useKnowledgeQA()
  const [branchingEnabled] = useKnowledgeQaBranching()
  const [isComparisonOpen, setIsComparisonOpen] = useState(false)
  const [leftThreadId, setLeftThreadId] = useState<string | null>(null)
  const [rightThreadId, setRightThreadId] = useState<string | null>(null)
  const [leftTurnId, setLeftTurnId] = useState<string | null>(null)
  const [rightTurnId, setRightTurnId] = useState<string | null>(null)
  const [loadingThreadIds, setLoadingThreadIds] = useState<string[]>([])
  const [failedThreadIds, setFailedThreadIds] = useState<string[]>([])
  const [branchingTurnId, setBranchingTurnId] = useState<string | null>(null)
  const [loadedTurnsByThread, setLoadedTurnsByThread] = useState<
    Record<string, ConversationTurn[]>
  >({})

  const currentTurns = useMemo(
    () => buildConversationTurns(messages, currentThreadId || "current"),
    [messages, currentThreadId]
  )
  const latestTurn =
    currentTurns.length > 0 ? currentTurns[currentTurns.length - 1] : null
  const historicalTurns = currentTurns.length > 1 ? currentTurns.slice(0, -1) : []

  const threadOptions = useMemo(() => {
    const options: ThreadOption[] = []
    const seenIds = new Set<string>()

    if (currentThreadId) {
      options.push({
        id: currentThreadId,
        label: "Current thread",
      })
      seenIds.add(currentThreadId)
    }

    for (const item of searchHistory) {
      const conversationId = item.conversationId
      if (!conversationId || conversationId.startsWith("local-")) continue
      if (seenIds.has(conversationId)) continue
      seenIds.add(conversationId)
      const label = item.query?.trim()
        ? `History: ${item.query.trim()}`
        : `History: ${conversationId}`
      options.push({
        id: conversationId,
        label: label.slice(0, 120),
      })
    }

    return options
  }, [currentThreadId, searchHistory])

  const threadLabelMap = useMemo(() => {
    const map = new Map<string, string>()
    for (const option of threadOptions) {
      map.set(option.id, option.label)
    }
    return map
  }, [threadOptions])

  useEffect(() => {
    if (!currentThreadId) return
    setLoadedTurnsByThread((previous) => ({
      ...previous,
      [currentThreadId]: currentTurns,
    }))
  }, [currentThreadId, currentTurns])

  useEffect(() => {
    if (threadOptions.length === 0) {
      setLeftThreadId(null)
      setRightThreadId(null)
      return
    }
    setLeftThreadId((previous) => {
      if (previous && threadOptions.some((option) => option.id === previous)) {
        return previous
      }
      return threadOptions[0].id
    })
    setRightThreadId((previous) => {
      if (previous && threadOptions.some((option) => option.id === previous)) {
        return previous
      }
      const alternative = threadOptions.find(
        (option) => option.id !== threadOptions[0].id
      )
      return alternative?.id || threadOptions[0].id
    })
  }, [threadOptions])

  const loadThreadTurns = useCallback(
    async (threadId: string) => {
      if (!threadId) return
      if (currentThreadId && threadId === currentThreadId) return
      if (loadedTurnsByThread[threadId]) return

      setLoadingThreadIds((previous) =>
        previous.includes(threadId) ? previous : [...previous, threadId]
      )
      setFailedThreadIds((previous) =>
        previous.includes(threadId)
          ? previous.filter((candidate) => candidate !== threadId)
          : previous
      )
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/chat/conversations/${threadId}/messages-with-context?include_rag_context=true`
        )
        if (!response.ok) {
          throw new Error(`Failed to load comparison thread ${threadId}`)
        }
        const rawMessages = await response.json()
        const normalizedMessages = normalizeRemoteMessages(rawMessages, threadId)
        const turns = buildConversationTurns(normalizedMessages, threadId)
        setLoadedTurnsByThread((previous) => ({
          ...previous,
          [threadId]: turns,
        }))
        setFailedThreadIds((previous) =>
          previous.filter((candidate) => candidate !== threadId)
        )
      } catch (error) {
        console.error("Failed to load comparison thread:", error)
        setFailedThreadIds((previous) =>
          previous.includes(threadId) ? previous : [...previous, threadId]
        )
      } finally {
        setLoadingThreadIds((previous) =>
          previous.filter((candidate) => candidate !== threadId)
        )
      }
    },
    [currentThreadId, loadedTurnsByThread]
  )

  useEffect(() => {
    if (leftThreadId) {
      void loadThreadTurns(leftThreadId)
    }
  }, [leftThreadId, loadThreadTurns])

  useEffect(() => {
    if (rightThreadId) {
      void loadThreadTurns(rightThreadId)
    }
  }, [rightThreadId, loadThreadTurns])

  const leftTurns = useMemo(
    () =>
      getTurnsForThread(
        leftThreadId,
        currentThreadId,
        currentTurns,
        loadedTurnsByThread
      ),
    [leftThreadId, currentThreadId, currentTurns, loadedTurnsByThread]
  )
  const rightTurns = useMemo(
    () =>
      getTurnsForThread(
        rightThreadId,
        currentThreadId,
        currentTurns,
        loadedTurnsByThread
      ),
    [rightThreadId, currentThreadId, currentTurns, loadedTurnsByThread]
  )

  useEffect(() => {
    if (leftTurns.length === 0) {
      setLeftTurnId(null)
      return
    }
    setLeftTurnId((previous) => {
      if (previous && leftTurns.some((turn) => turn.id === previous)) {
        return previous
      }
      return leftTurns[leftTurns.length - 1].id
    })
  }, [leftTurns])

  useEffect(() => {
    if (rightTurns.length === 0) {
      setRightTurnId(null)
      return
    }
    setRightTurnId((previous) => {
      if (previous && rightTurns.some((turn) => turn.id === previous)) {
        return previous
      }
      return rightTurns[rightTurns.length - 1].id
    })
  }, [rightTurns])

  useEffect(() => {
    if (!leftTurnId || !rightTurnId) return
    if (leftThreadId !== rightThreadId) return
    if (leftTurnId !== rightTurnId) return
    const alternative = rightTurns.find((turn) => turn.id !== leftTurnId)
    if (alternative) {
      setRightTurnId(alternative.id)
    }
  }, [leftThreadId, leftTurnId, rightThreadId, rightTurnId, rightTurns])

  const selectedLeftTurn =
    leftTurns.find((turn) => turn.id === leftTurnId) || leftTurns[leftTurns.length - 1]
  const selectedRightTurn =
    rightTurns.find((turn) => turn.id === rightTurnId) ||
    rightTurns[rightTurns.length - 1]

  const comparisonDraft =
    selectedLeftTurn && selectedRightTurn
      ? createComparisonDraft({
          left: {
            query: selectedLeftTurn.question,
            threadId: selectedLeftTurn.threadId,
            messageId: selectedLeftTurn.id,
            answer: selectedLeftTurn.answer,
            citationIndices: selectedLeftTurn.citationIndices,
          },
          right: {
            query: selectedRightTurn.question,
            threadId: selectedRightTurn.threadId,
            messageId: selectedRightTurn.id,
            answer: selectedRightTurn.answer,
            citationIndices: selectedRightTurn.citationIndices,
          },
        })
      : null
  const comparisonReady = comparisonDraft ? isComparisonReady(comparisonDraft) : false
  const isSelectedComparisonThreadLoading =
    (leftThreadId ? loadingThreadIds.includes(leftThreadId) : false) ||
    (rightThreadId ? loadingThreadIds.includes(rightThreadId) : false)
  const hasSelectedComparisonThreadFailure =
    (leftThreadId ? failedThreadIds.includes(leftThreadId) : false) ||
    (rightThreadId ? failedThreadIds.includes(rightThreadId) : false)

  const handleReuseQuestion = (question: string) => {
    setQuery(question)
    const searchInput = document.getElementById("knowledge-search-input")
    if (searchInput instanceof HTMLInputElement || searchInput instanceof HTMLTextAreaElement) {
      searchInput.focus()
      searchInput.setSelectionRange(searchInput.value.length, searchInput.value.length)
    }
  }

  const handleStartBranch = useCallback(
    async (messageId: string) => {
      if (branchingTurnId !== null) {
        return
      }
      setBranchingTurnId(messageId)
      try {
        await branchFromTurn(messageId)
      } finally {
        setBranchingTurnId(null)
      }
    },
    [branchFromTurn, branchingTurnId]
  )

  const handleCompareWithPrevious = useCallback(() => {
    if (!currentThreadId) return
    if (!latestTurn) return
    if (historicalTurns.length === 0) return

    const previousTurn = historicalTurns[historicalTurns.length - 1]
    setLeftThreadId(currentThreadId)
    setRightThreadId(currentThreadId)
    setLeftTurnId(previousTurn.id)
    setRightTurnId(latestTurn.id)
    setIsComparisonOpen(true)
  }, [currentThreadId, historicalTurns, latestTurn])

  const handleCompareTurnToCurrent = useCallback(
    (turnId: string) => {
      if (!currentThreadId) return
      if (!latestTurn) return

      setLeftThreadId(currentThreadId)
      setRightThreadId(currentThreadId)
      setLeftTurnId(turnId)
      setRightTurnId(latestTurn.id)
      setIsComparisonOpen(true)
    },
    [currentThreadId, latestTurn]
  )

  const hasComparisonWorkspace = threadOptions.length > 0
  const canCompareWithPrevious =
    Boolean(currentThreadId) && Boolean(latestTurn) && historicalTurns.length > 0
  const hasAdvancedComparisonWorkspace =
    hasComparisonWorkspace && (threadOptions.length > 1 || historicalTurns.length > 1)
  const showComparisonWorkspace = hasAdvancedComparisonWorkspace || isComparisonOpen
  if (historicalTurns.length === 0) {
    return null
  }

  return (
    <section
      className={cn("rounded-xl border border-border bg-muted/20", className)}
      aria-label="Conversation thread"
    >
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
        <MessageSquare className="w-4 h-4 text-text-muted" />
        <h2 className="text-sm font-semibold">
          Conversation Thread ({historicalTurns.length} prior turn
          {historicalTurns.length === 1 ? "" : "s"})
        </h2>
      </div>

      {historicalTurns.length > 0 ? (
        <div className="divide-y divide-border">
          {historicalTurns.map((turn, turnIndex) => (
            <article key={turn.id} className="px-4 py-3" role="article">
              <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
                <span>Question {turnIndex + 1}</span>
                {turn.timestamp ? <span>{formatTurnTimestamp(turn.timestamp)}</span> : null}
                <span className="inline-flex items-center gap-1 rounded bg-surface px-2 py-0.5">
                  <Sparkles className="w-3 h-3" />
                  {turn.sourceCount} source{turn.sourceCount === 1 ? "" : "s"}
                </span>
                {turn.citationCount > 0 ? (
                  <span className="inline-flex items-center gap-1 rounded bg-surface px-2 py-0.5">
                    {turn.citationCount} citation{turn.citationCount === 1 ? "" : "s"}
                  </span>
                ) : null}
              </div>

              <p className="mt-2 text-sm font-medium">{turn.question}</p>
              {turn.answer ? (
                <p className="mt-2 text-sm whitespace-pre-wrap leading-relaxed text-text-muted">
                  {truncatePreview(turn.answer, 220)}
                </p>
              ) : (
                <p className="mt-2 text-sm text-text-muted">No answer recorded.</p>
              )}

              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleReuseQuestion(turn.question)}
                  className="rounded-md border border-border bg-surface px-2 py-1 text-xs font-medium text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  Reuse Question
                </button>
                {branchingEnabled ? (
                  <button
                    type="button"
                    onClick={() => void handleStartBranch(turn.id)}
                    disabled={branchingTurnId !== null}
                    className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-xs font-medium text-text-subtle hover:bg-hover hover:text-text transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    <GitBranch className="w-3 h-3" />
                    {branchingTurnId === turn.id ? "Creating branch..." : "Ask a different follow-up"}
                  </button>
                ) : null}
                {canCompareWithPrevious && turnIndex === historicalTurns.length - 1 ? (
                  <button
                    type="button"
                    onClick={() => handleCompareTurnToCurrent(turn.id)}
                    className="rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/15 transition-colors"
                  >
                    Compare to current answer
                  </button>
                ) : null}
              </div>

              {turn.answer && turn.answer.length > 220 ? (
                <details className="mt-2 group">
                  <summary className="cursor-pointer text-xs text-primary hover:text-primaryStrong transition-colors">
                    Show full answer
                  </summary>
                  <p className="mt-2 text-sm whitespace-pre-wrap leading-relaxed">
                    {turn.answer}
                  </p>
                </details>
              ) : null}
            </article>
          ))}
        </div>
      ) : null}

      {showComparisonWorkspace && (
        <div className="border-t border-border px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">Compare answers</h3>
            {hasAdvancedComparisonWorkspace ? (
              <div className="flex flex-wrap items-center gap-2">
                {canCompareWithPrevious ? (
                  <button
                    type="button"
                    onClick={handleCompareWithPrevious}
                    className="rounded-md border border-primary/40 bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/15 transition-colors"
                  >
                    Compare to current answer
                  </button>
                ) : null}
                <button
                  type="button"
                  onClick={() => setIsComparisonOpen((previous) => !previous)}
                  className="rounded-md border border-border bg-surface px-2 py-1 text-xs font-medium text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  {isComparisonOpen ? "Hide comparison" : "Compare turns"}
                </button>
              </div>
            ) : null}
          </div>

          {isComparisonOpen && (
            <div className="mt-3 space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-1">
                  <label
                    htmlFor="knowledge-thread-compare-left-thread"
                    className="text-xs font-medium text-text-muted"
                  >
                    Left thread
                  </label>
                  <select
                    id="knowledge-thread-compare-left-thread"
                    value={leftThreadId || ""}
                    onChange={(event) => setLeftThreadId(event.target.value || null)}
                    className="w-full rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-text"
                  >
                    {threadOptions.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <label
                    htmlFor="knowledge-thread-compare-left-turn"
                    className="text-xs font-medium text-text-muted"
                  >
                    Left turn
                  </label>
                  <select
                    id="knowledge-thread-compare-left-turn"
                    value={leftTurnId || ""}
                    onChange={(event) => setLeftTurnId(event.target.value || null)}
                    className="w-full rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-text"
                  >
                    {leftTurns.map((turn) => (
                      <option key={turn.id} value={turn.id}>
                        {turn.question}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1">
                  <label
                    htmlFor="knowledge-thread-compare-right-thread"
                    className="text-xs font-medium text-text-muted"
                  >
                    Right thread
                  </label>
                  <select
                    id="knowledge-thread-compare-right-thread"
                    value={rightThreadId || ""}
                    onChange={(event) => setRightThreadId(event.target.value || null)}
                    className="w-full rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-text"
                  >
                    {threadOptions.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <label
                    htmlFor="knowledge-thread-compare-right-turn"
                    className="text-xs font-medium text-text-muted"
                  >
                    Right turn
                  </label>
                  <select
                    id="knowledge-thread-compare-right-turn"
                    value={rightTurnId || ""}
                    onChange={(event) => setRightTurnId(event.target.value || null)}
                    className="w-full rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-text"
                  >
                    {rightTurns.map((turn) => (
                      <option key={turn.id} value={turn.id}>
                        {turn.question}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {isSelectedComparisonThreadLoading ? (
                <p className="text-xs text-text-muted">Loading comparison thread...</p>
              ) : null}
              {!isSelectedComparisonThreadLoading && hasSelectedComparisonThreadFailure ? (
                <p className="text-xs text-warning">
                  Unable to load one of the selected comparison threads. Choose it again to retry.
                </p>
              ) : null}

              {comparisonReady ? (
                <div
                  role="region"
                  aria-label="Side-by-side query comparison"
                  className="grid gap-3 md:grid-cols-2"
                >
                  <article className="rounded-lg border border-border bg-surface p-3">
                    <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                      {threadLabelMap.get(comparisonDraft.left.threadId || "") || "Left thread"}
                    </h4>
                    <p className="mt-2 text-sm font-medium">
                      {comparisonDraft.left.query || "No query available"}
                    </p>
                    <p className="mt-2 text-sm whitespace-pre-wrap leading-relaxed text-text-muted">
                      {comparisonDraft.left.answer || "No answer available for this turn."}
                    </p>
                  </article>

                  <article className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                    <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                      {threadLabelMap.get(comparisonDraft.right.threadId || "") || "Right thread"}
                    </h4>
                    <p className="mt-2 text-sm font-medium">
                      {comparisonDraft.right.query || "No query available"}
                    </p>
                    <p className="mt-2 text-sm whitespace-pre-wrap leading-relaxed text-text-muted">
                      {comparisonDraft.right.answer || "No answer available for this turn."}
                    </p>
                  </article>
                </div>
              ) : (
                <p className="text-xs text-text-muted">
                  Select two complete turns to compare answers side by side.
                </p>
              )}
            </div>
          )}
        </div>
      )}
      {latestTurn === null && historicalTurns.length === 0 ? (
        <p className="px-4 py-3 text-xs text-text-muted">
          Run at least one search to build a conversation thread.
        </p>
      ) : null}
    </section>
  )
}
