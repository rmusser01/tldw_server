import React from "react"
import { Button, Spin } from "antd"
import { Sparkles } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { RagPinnedResult } from "@/utils/rag-format"
import type { QASearchResponse, QADocument } from "../hooks/useQASearch"
import { SearchEmptyState } from "../SearchTab/SearchEmptyState"
import { GeneratedAnswerCard } from "./GeneratedAnswerCard"
import { SourceChunksList } from "./SourceChunksList"

type QAResultsViewProps = {
  response: QASearchResponse | null
  loading: boolean
  hasAttemptedSearch: boolean
  timedOut: boolean
  query?: string
  pinnedResults: RagPinnedResult[]
  onRetry?: () => void
  /** Generate an answer from the shown sources; omit to hide the action. */
  onGenerateAnswer?: () => void
  answerLoading?: boolean
  answerError?: string | null
  onCopyAnswer: () => void
  onInsertAnswer: () => void
  onCopyChunk: (doc: QADocument) => void
  onInsertChunk: (doc: QADocument) => void
  onPinChunk: (doc: QADocument) => void
  onPreviewChunk: (doc: QADocument) => void
}

/**
 * Orchestrates the QA Search results display: source chunks first, then the
 * generated answer, which the user asks for (CC-10, owner decision [2]).
 */
export const QAResultsView: React.FC<QAResultsViewProps> = ({
  response,
  loading,
  hasAttemptedSearch,
  timedOut,
  query,
  pinnedResults,
  onRetry,
  onGenerateAnswer,
  answerLoading = false,
  answerError = null,
  onCopyAnswer,
  onInsertAnswer,
  onCopyChunk,
  onInsertChunk,
  onPinChunk,
  onPreviewChunk
}) => {
  const { t } = useTranslation(["sidepanel"])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spin size="default" />
        <span className="ml-2 text-sm text-text-muted">
          {t("sidepanel:qaSearch.searching", "Searching knowledge base...")}
        </span>
      </div>
    )
  }

  if (timedOut) {
    return <SearchEmptyState variant="timeout" onRetry={onRetry} />
  }

  if (!hasAttemptedSearch) {
    return <SearchEmptyState variant="initial" />
  }

  if (!response) {
    return <SearchEmptyState variant="no-results" />
  }

  const hasAnswer = !!response.generatedAnswer
  const hasDocs = response.documents.length > 0

  if (!hasAnswer && !hasDocs) {
    return <SearchEmptyState variant="no-results" />
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Errors */}
      {response.errors.length > 0 && (
        <div className="rounded-lg border border-warn/30 bg-warn/5 p-3">
          <p className="text-xs text-warn">
            {response.errors.join("; ")}
          </p>
        </div>
      )}

      {/* Source Chunks */}
      {hasDocs && (
        <SourceChunksList
          documents={response.documents}
          query={query}
          pinnedResults={pinnedResults}
          onCopy={onCopyChunk}
          onInsert={onInsertChunk}
          onPin={onPinChunk}
          onPreview={onPreviewChunk}
        />
      )}

      {/* Generated Answer, after the sources and only on request */}
      {hasAnswer ? (
        <GeneratedAnswerCard
          answer={response.generatedAnswer!}
          totalTime={response.totalTime}
          cacheHit={response.cacheHit}
          onCopy={onCopyAnswer}
          onInsert={onInsertAnswer}
        />
      ) : hasDocs && onGenerateAnswer ? (
        <div className="flex flex-col items-start gap-2">
          <Button
            size="small"
            onClick={onGenerateAnswer}
            disabled={answerLoading}
            aria-busy={answerLoading}
            icon={<Sparkles className="h-3 w-3" aria-hidden="true" />}
          >
            {answerLoading
              ? t("sidepanel:qaSearch.generatingAnswer", "Generating answer…")
              : t("sidepanel:qaSearch.generateAnswer", "Generate answer")}
          </Button>
          {answerError && (
            <p role="alert" className="text-xs text-danger">
              {answerError}
            </p>
          )}
        </div>
      ) : null}
    </div>
  )
}
