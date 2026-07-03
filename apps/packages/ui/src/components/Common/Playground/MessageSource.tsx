import { KnowledgeIcon } from "@/components/Option/Knowledge/KnowledgeIcon"
import { safeExternalUrl } from "@/utils/safe-external-url"
import { useTranslation } from "react-i18next"
import React from "react"

type Props = {
  source: {
    name?: string
    url?: string
    mode?: string
    type?: string
    score?: number
    relevance?: number
    chunk_id?: string
    strategy?: string
    retrieval_strategy?: string
    search_mode?: string
    source_type?: string
    rationale?: string
    reason?: string
    why_selected?: string
    pageContent?: string
    content?: string
    text?: string
    snippet?: string
    metadata?: {
      source?: string
      title?: string
      score?: number
      relevance?: number
      rerank_score?: number
      bm25_norm?: number
      chunk_id?: string
      search_mode?: string
      retrieval_strategy?: string
      reranking_strategy?: string
      source_type?: string
      type?: string
      reason?: string
      rationale?: string
      selection_reason?: string
      why_selected?: string
      page?: number
      loc?: {
        lines?: {
          from?: number
          to?: number
        }
      }
    }
  }
  onSourceClick?: (source: any) => void
  onSourceNavigate?: (source: any) => void
  onSourceDwell?: (source: any, dwellMs: number) => void
  onOpenKnowledgePanel?: () => void
}

export const MessageSource: React.FC<Props> = ({
  source,
  onSourceClick,
  onSourceNavigate,
  onSourceDwell,
  onOpenKnowledgePanel
}) => {
  const { t } = useTranslation("common")
  const detailsRef = React.useRef<HTMLDetailsElement | null>(null)
  const dwellStartRef = React.useRef<number | null>(null)
  const isKnowledge = source?.mode === "rag" || source?.mode === "chat"
  const label =
    source?.name ||
    source?.metadata?.source ||
    source?.metadata?.title ||
    source?.url ||
    t("sourceLabel", "Source")
  const content =
    source?.pageContent ||
    source?.content ||
    source?.text ||
    source?.snippet ||
    ""
  const url = source?.url
  const safeUrl = safeExternalUrl(url)
  const page = source?.metadata?.page
  const lineFrom = source?.metadata?.loc?.lines?.from
  const lineTo = source?.metadata?.loc?.lines?.to
  const isExpandable = Boolean(content)
  const resolveScoreLabel = React.useCallback((value: unknown): string | null => {
    if (typeof value !== "number" || !Number.isFinite(value)) return null
    if (value >= 0 && value <= 1) {
      return `${Math.round(value * 100)}%`
    }
    if (value > 1 && value <= 100) {
      return `${Math.round(value)}%`
    }
    return value.toFixed(2)
  }, [])
  const transparencyDetails = React.useMemo(
    () => {
      const metadata = source?.metadata || {}
      const details: Array<{ key: string; value: string }> = []
      const scoreLabel =
        resolveScoreLabel(source?.score) ||
        resolveScoreLabel(source?.relevance) ||
        resolveScoreLabel(metadata?.score) ||
        resolveScoreLabel(metadata?.relevance) ||
        resolveScoreLabel(metadata?.rerank_score) ||
        resolveScoreLabel(metadata?.bm25_norm)
      if (scoreLabel) {
        details.push({
          key: t("sourceWhyScore", "Relevance"),
          value: scoreLabel
        })
      }
      const chunkId =
        source?.chunk_id ||
        (source as any)?.chunkId ||
        metadata?.chunk_id ||
        (metadata as any)?.chunkId
      if (typeof chunkId === "string" && chunkId.trim().length > 0) {
        details.push({
          key: t("sourceWhyChunk", "Chunk"),
          value: chunkId.trim()
        })
      }
      const strategy =
        source?.strategy ||
        source?.retrieval_strategy ||
        source?.search_mode ||
        metadata?.retrieval_strategy ||
        metadata?.reranking_strategy ||
        metadata?.search_mode
      if (typeof strategy === "string" && strategy.trim().length > 0) {
        details.push({
          key: t("sourceWhyStrategy", "Strategy"),
          value: strategy.trim()
        })
      }
      const sourceType =
        source?.source_type ||
        source?.type ||
        metadata?.source_type ||
        metadata?.type
      if (typeof sourceType === "string" && sourceType.trim().length > 0) {
        details.push({
          key: t("sourceWhyType", "Source type"),
          value: sourceType.trim()
        })
      }
      const rationale =
        source?.rationale ||
        source?.reason ||
        source?.why_selected ||
        metadata?.selection_reason ||
        metadata?.rationale ||
        metadata?.reason ||
        metadata?.why_selected
      if (typeof rationale === "string" && rationale.trim().length > 0) {
        details.push({
          key: t("sourceWhyReason", "Reason"),
          value: rationale.trim()
        })
      }
      return details
    },
    [resolveScoreLabel, source, t]
  )

  const emitDwell = React.useCallback(() => {
    if (!onSourceDwell || dwellStartRef.current == null) return
    const dwellMs = Date.now() - dwellStartRef.current
    dwellStartRef.current = null
    onSourceDwell(source, dwellMs)
  }, [onSourceDwell, source])

  React.useEffect(() => {
    return () => {
      emitDwell()
    }
  }, [emitDwell])

  if (!isExpandable) {
    if (safeUrl) {
      return (
        <a
          href={safeUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => {
            onSourceNavigate && onSourceNavigate(source)
          }}
          className="inline-flex items-center rounded-md border border-border bg-surface2 px-2 py-1 text-caption text-text opacity-80 transition-shadow duration-200 ease-in-out hover:bg-surface hover:opacity-100 hover:shadow-md">
          <span className="text-caption">{label}</span>
        </a>
      )
    }

    return (
      <span className="inline-flex items-center rounded-md border border-border bg-surface2 px-2 py-1 text-caption text-text opacity-80">
        {label}
      </span>
    )
  }

  return (
    <details
      ref={detailsRef}
      className="w-full rounded-md border border-border bg-surface2 px-2 py-1"
      onToggle={(event) => {
        const detailsElement = event.currentTarget as HTMLDetailsElement
        if (detailsElement.open) {
          dwellStartRef.current = Date.now()
          return
        }
        emitDwell()
      }}>
      <summary
        onClick={() => {
          onSourceClick && onSourceClick(source)
        }}
        className="flex cursor-pointer items-center gap-2 text-caption text-text opacity-80 hover:opacity-100"
      >
        {isKnowledge && (
          <KnowledgeIcon type={source.type} className="h-3 w-3" />
        )}
        <span className="text-caption">{label}</span>
      </summary>
      <div className="mt-2 rounded-md border border-border bg-surface px-2 py-2 text-xs text-text-muted">
        <p className="whitespace-pre-wrap text-xs text-text-muted">{content}</p>
        {transparencyDetails.length > 0 && (
          <div className="mt-2 rounded-md border border-border bg-surface2 px-2 py-1.5">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-text-subtle">
              {t("sourceWhyTitle", "Why this source")}
            </p>
            <div className="flex flex-wrap gap-1.5 text-[11px] text-text-muted">
              {transparencyDetails.map((detail) => (
                <span
                  key={`${detail.key}:${detail.value}`}
                  className="rounded-md border border-border bg-surface px-1.5 py-0.5"
                >
                  <span className="font-medium">{detail.key}:</span>{" "}
                  <span>{detail.value}</span>
                </span>
              ))}
            </div>
          </div>
        )}
        {(page != null || lineFrom != null || url) && (
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-text-subtle">
            {page != null && (
              <span className="rounded-md border border-border bg-surface2 px-2 py-0.5">
                {`Page ${page}`}
              </span>
            )}
            {lineFrom != null && lineTo != null && (
              <span className="rounded-md border border-border bg-surface2 px-2 py-0.5">
                {`Line ${lineFrom} - ${lineTo}`}
              </span>
            )}
            {safeUrl && (
              <a
                href={safeUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => {
                  onSourceNavigate && onSourceNavigate(source)
                }}
                className="rounded-md border border-border bg-surface2 px-2 py-0.5 text-text-subtle hover:text-text"
              >
                {t("sourceOpen", "Open source")}
              </a>
            )}
          </div>
        )}
        {onOpenKnowledgePanel && (
          <div className="mt-2">
            <button
              type="button"
              onClick={onOpenKnowledgePanel}
              className="rounded-md border border-border bg-surface2 px-2 py-0.5 text-xs text-text-subtle hover:text-text"
            >
              {t("sourceOpenKnowledgePanel", "Open Search & Context")}
            </button>
          </div>
        )}
      </div>
    </details>
  )
}
