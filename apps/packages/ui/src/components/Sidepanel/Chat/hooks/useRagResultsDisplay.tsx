import React from "react"
import { Modal } from "antd"
import {
  formatRagResult,
  type RagCopyFormat,
  type RagPinnedResult
} from "@/utils/rag-format"
import { withFullMediaTextIfAvailable } from "@/components/Knowledge/hooks"
import { openExternalUrl } from "@/utils/safe-external-url"
import type { RagSettings } from "@/services/rag/unified-rag"
import type { MenuProps } from "antd"

export type RagResult = {
  content?: string
  text?: string
  chunk?: string
  metadata?: any
  score?: number
  relevance?: number
}

export type BatchResultGroup = {
  query: string
  results: RagResult[]
}

export const getResultText = (item: RagResult) =>
  item.content || item.text || item.chunk || ""

export const getResultTitle = (item: RagResult) =>
  item.metadata?.title || item.metadata?.source || item.metadata?.url || ""

export const getResultUrl = (item: RagResult) =>
  item.metadata?.url || item.metadata?.source || ""

export const getResultType = (item: RagResult) => item.metadata?.type || ""

export const getResultDate = (item: RagResult) =>
  item.metadata?.created_at || item.metadata?.date || item.metadata?.added_at

export const getResultScore = (item: RagResult) =>
  typeof item.score === "number"
    ? item.score
    : typeof item.relevance === "number"
      ? item.relevance
      : undefined

export const toPinnedResult = (item: RagResult): RagPinnedResult => {
  const snippet = getResultText(item).slice(0, 800)
  const title = getResultTitle(item)
  const url = getResultUrl(item)
  return {
    id: `${title || url || snippet.slice(0, 12)}-${item.metadata?.id || ""}`,
    title: title || undefined,
    source: item.metadata?.source || undefined,
    url: url || undefined,
    snippet,
    type: getResultType(item) || undefined
  }
}

export const formatScore = (score?: number) =>
  typeof score === "number" && Number.isFinite(score)
    ? score.toFixed(2)
    : null

export const formatDate = (value?: string | number) => {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(date)
}

export const highlightText = (text: string, query: string) => {
  const terms = query
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean)
  if (terms.length === 0) return text
  const escaped = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
  const regex = new RegExp(`(${escaped.join("|")})`, "gi")
  const parts = text.split(regex)
  const termSet = new Set(terms.map((term) => term.toLowerCase()))
  return parts.map((part, idx) =>
    termSet.has(part.toLowerCase()) ? (
      <mark key={`h-${idx}`} className="bg-warn/20 text-text">
        {part}
      </mark>
    ) : (
      <React.Fragment key={`t-${idx}`}>{part}</React.Fragment>
    )
  )
}

export interface UseRagResultsDisplayDeps {
  results: RagResult[]
  batchResults: BatchResultGroup[]
  ragPinnedResults: RagPinnedResult[] | undefined
  setRagPinnedResults: (value: RagPinnedResult[]) => void
  onInsert: (text: string) => void
  onAsk: (text: string, options?: { ignorePinnedResults?: boolean }) => void
  t: (key: string, fallback?: string) => string
}

export function useRagResultsDisplay(deps: UseRagResultsDisplayDeps) {
  const {
    results,
    batchResults,
    ragPinnedResults,
    setRagPinnedResults,
    onInsert,
    onAsk,
    t
  } = deps

  const [sortMode, setSortMode] = React.useState<"relevance" | "date" | "type">(
    "relevance"
  )
  const [previewItem, setPreviewItem] = React.useState<RagPinnedResult | null>(
    null
  )

  const sortResults = React.useCallback(
    (items: RagResult[]) => {
      if (sortMode === "type") {
        return [...items].sort((a, b) =>
          String(getResultType(a)).localeCompare(String(getResultType(b)))
        )
      }
      if (sortMode === "date") {
        return [...items].sort((a, b) => {
          const dateA = new Date(
            a.metadata?.created_at ||
              a.metadata?.date ||
              a.metadata?.added_at ||
              0
          ).getTime()
          const dateB = new Date(
            b.metadata?.created_at ||
              b.metadata?.date ||
              b.metadata?.added_at ||
              0
          ).getTime()
          return dateB - dateA
        })
      }
      return [...items].sort((a, b) => {
        const scoreA = getResultScore(a) ?? 0
        const scoreB = getResultScore(b) ?? 0
        return scoreB - scoreA
      })
    },
    [sortMode]
  )

  const copyResult = async (item: RagResult, format: RagCopyFormat) => {
    const pinned = toPinnedResult(item)
    await navigator.clipboard.writeText(formatRagResult(pinned, format))
  }

  const handleAsk = (item: RagResult) => {
    const pinned = toPinnedResult(item)
    if ((ragPinnedResults || []).length > 0) {
      Modal.confirm({
        title: t("sidepanel:rag.askConfirmTitle", "Ask about this item?") as string,
        content: t(
          "sidepanel:rag.askConfirmContent",
          "Pinned results will be ignored for this Ask."
        ) as string,
        okText: t("common:continue", "Continue") as string,
        cancelText: t("common:cancel", "Cancel") as string,
        onOk: () => onAsk(formatRagResult(pinned, "markdown"), { ignorePinnedResults: true })
      })
      return
    }
    onAsk(formatRagResult(pinned, "markdown"), { ignorePinnedResults: true })
  }

  const handleInsert = (item: RagResult) => {
    void (async () => {
      const pinned = toPinnedResult(item)
      const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
      onInsert(formatRagResult(resolvedPinned, "markdown"))
    })()
  }

  const handleOpen = (item: RagResult) => {
    const url = getResultUrl(item)
    if (!url) return
    openExternalUrl(String(url), "_blank")
  }

  const handlePin = (item: RagResult) => {
    const pinned = toPinnedResult(item)
    const existing = ragPinnedResults || []
    if (existing.some((result) => result.id === pinned.id)) return
    setRagPinnedResults([...existing, pinned])
  }

  const handleUnpin = (id: string) => {
    setRagPinnedResults((ragPinnedResults || []).filter((item) => item.id !== id))
  }

  const handleClearPins = () => setRagPinnedResults([])

  const copyMenu = (item: RagResult): MenuProps => ({
    items: [
      {
        key: "markdown",
        label: t("sidepanel:rag.copyMarkdown", "Copy as Markdown")
      },
      {
        key: "text",
        label: t("sidepanel:rag.copyText", "Copy as text")
      }
    ],
    onClick: ({ key }) => copyResult(item, key as RagCopyFormat)
  })

  return {
    sortMode,
    setSortMode,
    previewItem,
    setPreviewItem,
    sortResults,
    copyResult,
    handleAsk,
    handleInsert,
    handleOpen,
    handlePin,
    handleUnpin,
    handleClearPins,
    copyMenu,
  }
}
