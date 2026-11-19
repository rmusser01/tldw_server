import React from "react"
import { useTranslation } from "react-i18next"
import type { RagPinnedResult } from "@/utils/rag-format"
import type { QADocument } from "../hooks/useQASearch"
import { SourceChunkItem } from "./SourceChunkItem"
import { toPinnedResult } from "../hooks"
import { topChunkScore } from "./relevance-label"

type SourceChunksListProps = {
  documents: QADocument[]
  query?: string
  pinnedResults: RagPinnedResult[]
  onCopy: (doc: QADocument) => void
  onInsert: (doc: QADocument) => void
  onPin: (doc: QADocument) => void
  onPreview: (doc: QADocument) => void
}

type ChunkSortMode = "relevance" | "source"

const getChunkTitle = (doc: QADocument): string => {
  const title =
    (doc.metadata as Record<string, unknown>)?.title ??
    (doc.metadata as Record<string, unknown>)?.source
  return typeof title === "string" ? title.trim() : ""
}

const getChunkScore = (doc: QADocument): number => {
  const value = doc.score ?? doc.relevance
  return typeof value === "number" && Number.isFinite(value) ? value : 0
}

/**
 * Scrollable list of retrieved source chunks with plain relevance labels.
 */
export const SourceChunksList: React.FC<SourceChunksListProps> = ({
  documents,
  query,
  pinnedResults,
  onCopy,
  onInsert,
  onPin,
  onPreview
}) => {
  const { t } = useTranslation(["sidepanel"])
  const [sortMode, setSortMode] = React.useState<ChunkSortMode>("relevance")

  if (documents.length === 0) return null

  const sortedDocuments = React.useMemo(() => {
    const decorated = documents.map((doc, originalIndex) => ({
      doc,
      originalIndex,
      title: getChunkTitle(doc),
      score: getChunkScore(doc)
    }))

    decorated.sort((a, b) => {
      if (sortMode === "source") {
        const byTitle = a.title.localeCompare(b.title, undefined, {
          sensitivity: "base"
        })
        if (byTitle !== 0) return byTitle
        if (b.score !== a.score) return b.score - a.score
        return a.originalIndex - b.originalIndex
      }

      if (b.score !== a.score) return b.score - a.score
      const byTitle = a.title.localeCompare(b.title, undefined, {
        sensitivity: "base"
      })
      if (byTitle !== 0) return byTitle
      return a.originalIndex - b.originalIndex
    })

    return decorated.map((item) => item.doc)
  }, [documents, sortMode])
  const topScore = topChunkScore(documents)

  const isPinned = (doc: QADocument): boolean => {
    const ragResult = {
      content: doc.content || doc.text || doc.chunk || "",
      metadata: {
        ...doc.metadata,
        ...(doc.id !== undefined ? { id: doc.id } : {}),
        ...(doc.media_id !== undefined ? { media_id: doc.media_id } : {})
      },
      score: doc.score,
      relevance: doc.relevance
    }
    const pinned = toPinnedResult(ragResult)
    return pinnedResults.some((p) => p.id === pinned.id)
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wide">
          {t("sidepanel:qaSearch.sourceChunks", "Source Chunks ({{count}})", {
            count: documents.length
          })}
        </h4>
        <div
          className="inline-flex items-center rounded-md border border-border bg-surface2 p-0.5"
          role="group"
          aria-label={t("sidepanel:qaSearch.sort.label", "Sort source chunks")}
        >
          <button
            type="button"
            onClick={() => setSortMode("relevance")}
            aria-pressed={sortMode === "relevance"}
            className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
              sortMode === "relevance"
                ? "bg-accent text-white"
                : "text-text-muted hover:text-text"
            }`}
          >
            {t("sidepanel:qaSearch.sort.relevance", "Relevance")}
          </button>
          <button
            type="button"
            onClick={() => setSortMode("source")}
            aria-pressed={sortMode === "source"}
            className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
              sortMode === "source"
                ? "bg-accent text-white"
                : "text-text-muted hover:text-text"
            }`}
          >
            {t("sidepanel:qaSearch.sort.source", "Source")}
          </button>
        </div>
      </div>
      <div
        className="flex flex-col gap-2 max-h-[350px] overflow-y-auto"
        role="list"
        aria-label={t(
          "sidepanel:qaSearch.sourceChunksList",
          "Retrieved source chunks"
        )}
      >
        {sortedDocuments.map((doc, idx) => (
          <div key={`chunk-${idx}`} role="listitem">
            <SourceChunkItem
              document={doc}
              index={idx}
              query={query}
              onCopy={onCopy}
              onInsert={onInsert}
              onPin={onPin}
              onPreview={onPreview}
              isPinned={isPinned(doc)}
              topScore={topScore}
            />
          </div>
        ))}
      </div>
    </div>
  )
}
