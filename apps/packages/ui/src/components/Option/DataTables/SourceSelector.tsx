import React, { useState, useCallback, useMemo, useEffect, useRef } from "react"
import {
  Button,
  Card,
  Empty,
  Input,
  List,
  Segmented,
  Spin,
  Tag,
  message
} from "antd"
import { MessageSquare, FileText, Search, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { useDataTablesStore } from "@/store/data-tables"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { DataTableSource, DataTableSourceType } from "@/types/data-tables"
import { StatePanel } from "@/components/ui/state"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"

const DOCUMENT_MEDIA_TYPES = new Set([
  "document",
  "pdf",
  "ebook",
  "email",
  "code",
  "html"
])

const extractMediaItems = (response: unknown): unknown[] => {
  if (!response) return []
  if (Array.isArray(response)) return response
  if (Array.isArray((response as { items?: unknown[] }).items)) {
    return (response as { items: unknown[] }).items
  }
  if (Array.isArray((response as { results?: unknown[] }).results)) {
    return (response as { results: unknown[] }).results
  }
  if (Array.isArray((response as { data?: unknown[] }).data)) {
    return (response as { data: unknown[] }).data
  }
  if (Array.isArray((response as { media?: unknown[] }).media)) {
    return (response as { media: unknown[] }).media
  }
  return []
}

/**
 * SourceSelector
 *
 * Component for selecting data sources (chats, documents, RAG queries)
 * to use for table generation.
 */
export const SourceSelector: React.FC = () => {
  const { t } = useTranslation(["dataTables", "common"])

  // Store state
  const selectedSources = useDataTablesStore((s) => s.selectedSources)
  const activeSourceType = useDataTablesStore((s) => s.activeSourceType)
  const sourceSearchQuery = useDataTablesStore((s) => s.sourceSearchQuery)

  // Store actions
  const addSource = useDataTablesStore((s) => s.addSource)
  const removeSource = useDataTablesStore((s) => s.removeSource)
  const setActiveSourceType = useDataTablesStore((s) => s.setActiveSourceType)
  const setSourceSearchQuery = useDataTablesStore((s) => s.setSourceSearchQuery)

  const [ragQuery, setRagQuery] = useState("")

  // Source type options
  const sourceTypeOptions = useMemo(
    () => [
      {
        value: "chat" as DataTableSourceType,
        label: (
          <span className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            {t("dataTables:sourceTypes.chat", "Chats")}
          </span>
        )
      },
      {
        value: "document" as DataTableSourceType,
        label: (
          <span className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            {t("dataTables:sourceTypes.document", "Documents")}
          </span>
        )
      },
      {
        value: "rag_query" as DataTableSourceType,
        label: (
          <span className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            {t("dataTables:sourceTypes.rag", "RAG Search")}
          </span>
        )
      }
    ],
    [t]
  )

  const sourcesQuery = useQuery<DataTableSource[]>({
    queryKey: ["dataTables", "sources", activeSourceType, sourceSearchQuery],
    enabled: activeSourceType !== "rag_query",
    staleTime: 30 * 1000,
    queryFn: async ({ signal }) => {
      if (activeSourceType === "chat") {
        const chats = await tldwClient.listChats(
          {
            limit: 50,
            search: sourceSearchQuery || undefined
          },
          { signal }
        )
        return chats.map((chat) => ({
          type: "chat" as const,
          id: chat.id,
          title: chat.title || `Chat ${chat.id}`,
          snippet: chat.topic_label || undefined
        }))
      }
      if (activeSourceType === "document") {
        const mediaTypes = Array.from(DOCUMENT_MEDIA_TYPES)
        const response = sourceSearchQuery
          ? await tldwClient.searchMedia(
              {
                query: sourceSearchQuery,
                fields: ["title", "content"],
                media_types: mediaTypes
              },
              { page: 1, results_per_page: 50 },
              { signal }
            )
          : await tldwClient.listMedia(
              { page: 1, results_per_page: 50 },
              { signal }
            )

        const items = extractMediaItems(response)
        return items
          .map((item) => {
            if (!item || typeof item !== "object") return null
            const record = item as Record<string, unknown>
            const rawType = String(record.type ?? record.media_type ?? "").toLowerCase()
            if (rawType && !DOCUMENT_MEDIA_TYPES.has(rawType)) {
              return null
            }
            const title =
              (typeof record.title === "string" && record.title) ||
              (typeof record.name === "string" && record.name) ||
              `Document ${record.id ?? ""}`
            const snippet =
              (typeof record.content_preview === "string" && record.content_preview) ||
              (typeof record.description === "string" && record.description) ||
              (typeof record.summary === "string" && record.summary) ||
              (rawType ? rawType.toUpperCase() : undefined)
            return {
              type: "document" as const,
              id: String(record.id ?? ""),
              title,
              snippet
            }
          })
          .filter(Boolean) as DataTableSource[]
      }
      return []
    }
  })

  const lastErrorAtRef = useRef(0)

  useEffect(() => {
    if (!sourcesQuery.isError || !sourcesQuery.error) return
    if (sourcesQuery.errorUpdatedAt === lastErrorAtRef.current) return
    lastErrorAtRef.current = sourcesQuery.errorUpdatedAt

    const error = sourcesQuery.error
    const isAbortError =
      error instanceof Error &&
      (error.name === "AbortError" ||
        error.message.toLowerCase().includes("abort"))
    if (isAbortError) {
      return
    }
    console.error("[SourceSelector] Failed to fetch items", {
      error,
      activeSourceType,
      sourceSearchQuery
    })
    message.error(t("dataTables:fetchError", "Failed to load items"))
  }, [
    activeSourceType,
    sourceSearchQuery,
    sourcesQuery.error,
    sourcesQuery.errorUpdatedAt,
    sourcesQuery.isError,
    t
  ])

  const availableItems =
    activeSourceType === "rag_query" ? [] : sourcesQuery.data ?? []
  const sourcesError =
    activeSourceType !== "rag_query" && sourcesQuery.isError
      ? sourcesQuery.error instanceof Error
        ? sourcesQuery.error.message
        : t("dataTables:sourceLoadErrorFallback", "Failed to load sources")
      : null
  const activeSourceTypeLabel =
    activeSourceType === "chat"
      ? t("dataTables:sourceTypes.chat", "Chats")
      : t("dataTables:sourceTypes.document", "Documents")
  const sourceDiagnostics = useMemo(() => {
    if (!sourcesError) return []
    return [
      {
        label: t("dataTables:sourceLoadErrorTypeLabel", "Source type"),
        value: activeSourceTypeLabel
      },
      ...(sourceSearchQuery.trim()
        ? [
            {
              label: t("dataTables:sourceLoadErrorSearchLabel", "Search"),
              value: sourceSearchQuery.trim()
            }
          ]
        : []),
      {
        label: t("dataTables:sourceLoadErrorDetailsLabel", "Details"),
        value: sanitizeServerErrorMessage(
          sourcesError,
          t("dataTables:sourceLoadErrorFallback", "Failed to load sources")
        )
      }
    ]
  }, [activeSourceTypeLabel, sourceSearchQuery, sourcesError, t])
  const loading =
    activeSourceType !== "rag_query" &&
    (sourcesQuery.isLoading || sourcesQuery.isFetching)

  // Handle RAG query submission
  const handleRagSearch = useCallback(() => {
    const trimmedQuery = ragQuery.trim()
    if (!trimmedQuery) return

    const source: DataTableSource = {
      type: "rag_query",
      id: trimmedQuery,
      title: `RAG: "${trimmedQuery}"`,
      snippet: t("dataTables:ragQuerySnippet", "Knowledge base search query")
    }
    addSource(source)
    setRagQuery("")
    message.success(t("dataTables:ragAdded", "RAG query added as source"))
  }, [addSource, ragQuery, setRagQuery, t])

  // Check if source is selected
  const isSelected = (id: string) => selectedSources.some((s) => s.id === id)

  return (
    <div className="space-y-4">
      {/* Selected sources */}
      {selectedSources.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-text mb-2">
            {t("dataTables:selectedSources", "Selected Sources")} ({selectedSources.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {selectedSources.map((source) => (
              <Tag
                key={source.id}
                closable
                onClose={() => removeSource(source.id)}
                className="flex items-center gap-1"
              >
                {source.type === "chat" && <MessageSquare className="h-3 w-3" />}
                {source.type === "document" && <FileText className="h-3 w-3" />}
                {source.type === "rag_query" && <Search className="h-3 w-3" />}
                <span className="max-w-[200px] truncate">{source.title}</span>
              </Tag>
            ))}
          </div>
        </div>
      )}

      {/* Source type selector */}
      <Segmented
        value={activeSourceType}
        onChange={(value) => setActiveSourceType(value as DataTableSourceType)}
        options={sourceTypeOptions}
        block
      />

      {/* Search or RAG input */}
      {activeSourceType === "rag_query" ? (
        <div className="flex gap-2">
          <Input
            placeholder={t("dataTables:ragPlaceholder", "Enter a search query for your knowledge base...")}
            value={ragQuery}
            onChange={(e) => setRagQuery(e.target.value)}
            onPressEnter={handleRagSearch}
            prefix={<Search className="h-4 w-4 text-text-subtle" />}
          />
          <Button
            type="primary"
            icon={<Plus className="h-4 w-4" />}
            onClick={handleRagSearch}
            disabled={!ragQuery.trim()}
          >
            {t("common:add", "Add")}
          </Button>
        </div>
      ) : (
        <Input
          placeholder={t("dataTables:searchPlaceholder", "Search...")}
          value={sourceSearchQuery}
          onChange={(e) => setSourceSearchQuery(e.target.value)}
          prefix={<Search className="h-4 w-4 text-text-subtle" />}
          allowClear
        />
      )}

      {/* Available items list */}
      {activeSourceType !== "rag_query" && (
        <Card className="max-h-[300px] overflow-y-auto">
          {loading ? (
            <div className="flex justify-center py-8">
              <Spin />
            </div>
          ) : sourcesError ? (
            <StatePanel
              state="unavailable"
              title={t(
                "dataTables:sourceLoadErrorTitle",
                "Data sources could not load"
              )}
              message={t(
                "dataTables:sourceLoadErrorBody",
                "Try again after checking the server connection. Your selected sources are preserved."
              )}
              diagnostics={sourceDiagnostics}
              primaryAction={{
                label: t("common:tryAgain", "Try again"),
                onClick: () => {
                  void sourcesQuery.refetch()
                },
                loading
              }}
              data-testid="data-tables-source-load-recovery"
            />
          ) : availableItems.length === 0 ? (
            <Empty
              description={t(
                "dataTables:noItemsFound",
                "No items found. Try a different search."
              )}
            />
          ) : (
            <List
              dataSource={availableItems}
              renderItem={(item) => (
                <List.Item
                  className={`cursor-pointer hover:bg-surface rounded-lg transition-colors ${
                    isSelected(item.id)
                      ? "bg-primary/10 border-l-2 border-primary"
                      : ""
                  }`}
                  onClick={() => {
                    if (isSelected(item.id)) {
                      removeSource(item.id)
                    } else {
                      addSource(item)
                    }
                  }}
                >
                  <List.Item.Meta
                    avatar={
                      item.type === "chat" ? (
                        <MessageSquare className="h-5 w-5 text-text-subtle" />
                      ) : (
                        <FileText className="h-5 w-5 text-text-subtle" />
                      )
                    }
                    title={
                      <span className="text-text">
                        {item.title}
                      </span>
                    }
                    description={
                      item.snippet && (
                        <span className="text-xs text-text-muted line-clamp-1">
                          {item.snippet}
                        </span>
                      )
                    }
                  />
                  {isSelected(item.id) ? (
                    <Tag color="blue">{t("common:selected", "Selected")}</Tag>
                  ) : (
                    <Button size="small" type="text" icon={<Plus className="h-4 w-4" />}>
                      {t("common:add", "Add")}
                    </Button>
                  )}
                </List.Item>
              )}
            />
          )}
        </Card>
      )}

      {/* Tip for RAG */}
      {activeSourceType === "rag_query" && (
        <p className="text-sm text-text-muted">
          {t(
            "dataTables:ragTip",
            "Enter search queries to find relevant content in your knowledge base. Each query will be added as a separate source."
          )}
        </p>
      )}
    </div>
  )
}
