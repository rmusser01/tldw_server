import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Alert, Button, Modal, Select, Space, Switch, Table, Tag, Upload, message } from "antd"
import type { RcFile } from "antd/es/upload/interface"
import { UploadCloud } from "lucide-react"
import { useTranslation } from "react-i18next"
import { fetchWatchlistSources, importOpml } from "@/services/watchlists"
import type { SourcesImportResponse, WatchlistGroup, WatchlistTag } from "@/types/watchlists"
import {
  buildWatchlistsModalChrome,
  useWatchlistsViewport,
  WatchlistsHelpTooltip
} from "../shared"
import { buildOpmlPreflightSummary, type OpmlPreflightItem, type OpmlPreflightStatus } from "./opml-preflight"

const EXISTING_URL_LOOKUP_PAGE_SIZE = 200
const EXISTING_URL_LOOKUP_MAX_PAGES = 10
const PREVIEW_ROW_LIMIT = 200

interface SourcesBulkImportProps {
  open: boolean
  onClose: () => void
  groups: WatchlistGroup[]
  tags: WatchlistTag[]
  defaultGroupId?: number | null
  watchlistId?: number | null
  onImported: () => void
}

const PREVIEW_STATUS_COLOR: Record<OpmlPreflightStatus, string> = {
  ready: "green",
  duplicate_existing: "orange",
  duplicate_file: "gold",
  missing_url: "red",
  invalid_url: "red"
}

type ImportFailureReasonCode =
  | "duplicate_existing"
  | "duplicate_file"
  | "missing_url"
  | "invalid_url"
  | "auth"
  | "timeout"
  | "network"
  | "import_error"

interface ImportFailureItem {
  name?: string | null
  url: string
  status: string
  error?: string | null
  reasonCode: ImportFailureReasonCode
}

const escapeXml = (value: string): string =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;")

const inferImportFailureReasonCode = (error: string | null | undefined): ImportFailureReasonCode => {
  const normalized = String(error || "").toLowerCase()
  if (normalized.includes("duplicate")) return "duplicate_existing"
  if (normalized.includes("missing") && normalized.includes("url")) return "missing_url"
  if (normalized.includes("invalid") && normalized.includes("url")) return "invalid_url"
  if (
    normalized.includes("401") ||
    normalized.includes("403") ||
    normalized.includes("unauthorized") ||
    normalized.includes("forbidden")
  ) {
    return "auth"
  }
  if (normalized.includes("timeout") || normalized.includes("timed out")) return "timeout"
  if (
    normalized.includes("failed to fetch") ||
    normalized.includes("networkerror") ||
    normalized.includes("connection")
  ) {
    return "network"
  }
  return "import_error"
}

const isRetryableFailure = (reasonCode: ImportFailureReasonCode): boolean =>
  reasonCode !== "duplicate_existing" &&
  reasonCode !== "duplicate_file" &&
  reasonCode !== "missing_url" &&
  reasonCode !== "invalid_url"

const buildOpmlFromFailureItems = (items: ImportFailureItem[]): string => {
  const outlines = items
    .map((item) => {
      const title = item.name && item.name.trim().length > 0 ? item.name : item.url
      return `    <outline text="${escapeXml(title)}" xmlUrl="${escapeXml(item.url)}" />`
    })
    .join("\n")
  return `<?xml version="1.0" encoding="UTF-8"?>\n<opml version="2.0">\n  <head>\n    <title>Failed Watchlists Import Retry</title>\n  </head>\n  <body>\n${outlines}\n  </body>\n</opml>\n`
}

const downloadText = (content: string, filename: string, mimeType: string): void => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

const toFileFromUploadInput = (file: RcFile | File | { originFileObj?: File | RcFile }): File | null => {
  if (file instanceof File) return file
  if ("originFileObj" in file && file.originFileObj instanceof File) return file.originFileObj
  if (typeof (file as File).text === "function") return file as File
  return null
}

export const SourcesBulkImport: React.FC<SourcesBulkImportProps> = ({
  open,
  onClose,
  groups,
  tags,
  defaultGroupId,
  watchlistId,
  onImported
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [active, setActive] = useState(true)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(defaultGroupId ?? null)
  const [importing, setImporting] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preflight, setPreflight] = useState<ReturnType<typeof buildOpmlPreflightSummary> | null>(null)
  const [existingUrls, setExistingUrls] = useState<string[]>([])
  const [existingUrlsLoading, setExistingUrlsLoading] = useState(false)
  const [existingUrlsLoaded, setExistingUrlsLoaded] = useState(false)
  const [result, setResult] = useState<SourcesImportResponse | null>(null)
  const [retryingFailedOnly, setRetryingFailedOnly] = useState(false)
  const { isConstrained } = useWatchlistsViewport()
  const modalChrome = buildWatchlistsModalChrome(isConstrained, 900)
  const existingUrlsRequestRef = useRef(0)

  const resetState = useCallback(() => {
    setActive(true)
    setSelectedTags([])
    setSelectedGroupId(defaultGroupId ?? null)
    setImporting(false)
    setSelectedFile(null)
    setPreflight(null)
    setExistingUrls([])
    setExistingUrlsLoading(false)
    setExistingUrlsLoaded(false)
    setResult(null)
    setRetryingFailedOnly(false)
  }, [defaultGroupId])

  const loadExistingUrls = useCallback(async (): Promise<string[]> => {
    const requestId = existingUrlsRequestRef.current + 1
    existingUrlsRequestRef.current = requestId
    setExistingUrlsLoading(true)
    setExistingUrlsLoaded(false)
    try {
      const urls: string[] = []
      let page = 1
      while (page <= EXISTING_URL_LOOKUP_MAX_PAGES) {
        const response = await fetchWatchlistSources({
          watchlist_id: watchlistId ?? undefined,
          page,
          size: EXISTING_URL_LOOKUP_PAGE_SIZE
        })
        const pageItems = Array.isArray(response.items) ? response.items : []
        urls.push(...pageItems.map((item) => item.url).filter(Boolean))
        if (
          pageItems.length < EXISTING_URL_LOOKUP_PAGE_SIZE ||
          urls.length >= Number(response.total || 0)
        ) {
          break
        }
        page += 1
      }
      if (existingUrlsRequestRef.current !== requestId) return []
      setExistingUrls(urls)
      setExistingUrlsLoaded(true)
      return urls
    } catch (err) {
      if (existingUrlsRequestRef.current !== requestId) return []
      console.error("Failed to load existing sources for import preview:", err)
      setExistingUrlsLoaded(true)
      message.warning(
        t(
          "watchlists:sources.importPreviewLookupError",
          "Could not load existing feeds for duplicate checks."
        )
      )
      return []
    } finally {
      if (existingUrlsRequestRef.current === requestId) {
        setExistingUrlsLoading(false)
      }
    }
  }, [t, watchlistId])

  useEffect(() => {
    if (!open) {
      existingUrlsRequestRef.current += 1
      resetState()
      return
    }
    resetState()
    void loadExistingUrls()
  }, [loadExistingUrls, open, resetState])

  const summary = useMemo(() => {
    if (!result?.items?.length) return null
    const created = result.items.filter((item) => item.status === "created").length
    const skipped = result.items.filter((item) => item.status === "skipped").length
    const errors = result.items.filter((item) => item.status === "error").length
    return { created, skipped, errors }
  }, [result])

  const canCommitImport = Boolean(
    selectedFile &&
    preflight &&
    preflight.ready > 0 &&
    !preflight.parseError &&
    !importing
  )

  const handlePreflight = useCallback(async (file: File) => {
    setSelectedFile(file)
    setResult(null)
    try {
      const opml = await file.text()
      const urls = existingUrlsLoaded ? existingUrls : await loadExistingUrls()
      const preview = buildOpmlPreflightSummary(opml, { existingUrls: urls })
      setPreflight(preview)
    } catch (err) {
      console.error("OPML preview failed:", err)
      setPreflight(null)
      message.error(
        t("watchlists:sources.importPreviewError", "Failed to preview OPML file")
      )
    }
  }, [existingUrls, existingUrlsLoaded, loadExistingUrls, t])

  const handleCommitImport = async () => {
    if (!selectedFile || !preflight) return
    if (preflight.parseError || preflight.ready === 0) {
      message.warning(
        t(
          "watchlists:sources.importPreviewNothingToImport",
          "No valid feed entries to import."
        )
      )
      return
    }

    setImporting(true)
    try {
      const response = await importOpml(selectedFile, {
        active,
        tags: selectedTags,
        group_id: selectedGroupId ?? undefined,
        watchlist_id: watchlistId ?? undefined
      })
      setResult(response)
      onImported()
      message.success(t("watchlists:sources.imported", "OPML imported"))
    } catch (err) {
      console.error("OPML import failed:", err)
      message.error(t("watchlists:sources.importError", "Failed to import OPML"))
    } finally {
      setImporting(false)
    }
  }

  const uploadProps = {
    accept: ".opml,.xml",
    multiple: false,
    showUploadList: false,
    beforeUpload: async (file: RcFile | { originFileObj?: RcFile }) => {
      const normalizedFile = toFileFromUploadInput(file)
      if (!normalizedFile) {
        message.error(
          t("watchlists:sources.importPreviewError", "Failed to preview OPML file")
        )
        return false
      }
      await handlePreflight(normalizedFile)
      return false
    }
  }

  const errorItems = (result?.items || []).filter((item) => item.status === "error")
  const failedItems = useMemo<ImportFailureItem[]>(
    () =>
      errorItems.map((item) => ({
        name: item.name,
        url: item.url,
        status: item.status,
        error: item.error,
        reasonCode: inferImportFailureReasonCode(item.error)
      })),
    [errorItems]
  )
  const retryableFailedItems = useMemo(
    () => failedItems.filter((item) => isRetryableFailure(item.reasonCode)),
    [failedItems]
  )

  const handleRetryFailedOnly = async () => {
    if (!selectedFile || retryableFailedItems.length === 0) {
      message.warning(
        t(
          "watchlists:sources.importRetryFailedNone",
          "No retryable failed feeds."
        )
      )
      return
    }

    setRetryingFailedOnly(true)
    try {
      const retryOpml = buildOpmlFromFailureItems(retryableFailedItems)
      const retryFile = new File(
        [retryOpml],
        `watchlists_failed_retry_${Date.now()}.opml`,
        { type: "text/xml" }
      )
      const retryResult = await importOpml(retryFile, {
        active,
        tags: selectedTags,
        group_id: selectedGroupId ?? undefined,
        watchlist_id: watchlistId ?? undefined
      })
      const retriedUrlSet = new Set(retryableFailedItems.map((item) => item.url))
      setResult((previous) => {
        if (!previous?.items?.length) return retryResult
        const retained = previous.items.filter(
          (item) => !(item.status === "error" && retriedUrlSet.has(item.url))
        )
        const nextItems = Array.isArray(retryResult.items) ? retryResult.items : []
        return {
          ...previous,
          items: [...retained, ...nextItems]
        }
      })
      onImported()
      message.success(
        t(
          "watchlists:sources.importRetryFailedSuccess",
          "Retried {{count}} failed feed{{plural}}.",
          {
            count: retryableFailedItems.length,
            plural: retryableFailedItems.length === 1 ? "" : "s"
          }
        )
      )
    } catch (err) {
      console.error("Retry failed-only import failed:", err)
      message.error(
        t(
          "watchlists:sources.importRetryFailedError",
          "Failed to retry failed feeds."
        )
      )
    } finally {
      setRetryingFailedOnly(false)
    }
  }

  const handleExportFailedCsv = () => {
    if (failedItems.length === 0) return
    const rows = [
      ["name", "url", "status", "reason_code", "error"].join(","),
      ...failedItems.map((item) => {
        const name = String(item.name || "")
        const url = String(item.url || "")
        const status = String(item.status || "")
        const reasonCode = String(item.reasonCode || "")
        const error = String(item.error || "")
        const escaped = [name, url, status, reasonCode, error].map((value) => {
          if (/[",\n]/.test(value)) return `"${value.replace(/"/g, "\"\"")}"`
          return value
        })
        return escaped.join(",")
      })
    ]
    downloadText(rows.join("\n"), `watchlists_import_failed_${Date.now()}.csv`, "text/csv;charset=utf-8")
  }

  const handleExportFailedJson = () => {
    if (failedItems.length === 0) return
    downloadText(
      JSON.stringify(failedItems, null, 2),
      `watchlists_import_failed_${Date.now()}.json`,
      "application/json;charset=utf-8"
    )
  }

  const renderPreflightStatus = (status: OpmlPreflightStatus) => (
    <Tag color={PREVIEW_STATUS_COLOR[status]}>
      {t(
        `watchlists:sources.importPreviewStatus.${status}`,
        status.replace(/_/g, " ")
      )}
    </Tag>
  )

  const renderPreflightPreview = () => {
    if (!preflight || preflight.items.length === 0) return null
    const items = preflight.items.slice(0, PREVIEW_ROW_LIMIT)

    if (isConstrained) {
      return (
        <div
          className="space-y-2"
          data-testid="opml-preflight-constrained-list"
        >
          {items.map((item, idx) => (
            <div
              key={`${item.url}-${idx}`}
              className="rounded-md border border-border bg-surface p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">
                    {item.name || item.url || "-"}
                  </div>
                  <div className="break-all text-xs text-text-muted">
                    {item.url || "-"}
                  </div>
                </div>
                <div className="shrink-0">
                  {renderPreflightStatus(item.status)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )
    }

    return (
      <Table<OpmlPreflightItem>
        dataSource={items}
        rowKey={(item, idx) => `${item.url}-${idx}`}
        pagination={false}
        size="small"
        columns={[
          {
            title: t("watchlists:sources.columns.name", "Name"),
            dataIndex: "name",
            key: "name",
            render: (name: string, record) => name || record.url || "-"
          },
          {
            title: t("watchlists:sources.columns.url", "URL"),
            dataIndex: "url",
            key: "url",
            ellipsis: true,
            render: (url: string) => url || "-"
          },
          {
            title: t("watchlists:sources.importPreviewStatusTitle", "Preflight Status"),
            dataIndex: "status",
            key: "status",
            width: 180,
            render: (status: OpmlPreflightStatus) => renderPreflightStatus(status)
          }
        ]}
      />
    )
  }

  const renderFailedItems = () => {
    if (failedItems.length === 0) return null

    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm text-text-muted">
            {t(
              "watchlists:sources.importFailedCount",
              "{{count}} failed feed{{plural}} ready for recovery actions.",
              {
                count: failedItems.length,
                plural: failedItems.length === 1 ? "" : "s"
              }
            )}
          </div>
          <Space wrap>
            <Button
              onClick={handleRetryFailedOnly}
              disabled={retryableFailedItems.length === 0}
              loading={retryingFailedOnly}
            >
              {t("watchlists:sources.importRetryFailed", "Retry failed only")}
            </Button>
            <Button onClick={handleExportFailedCsv}>
              {t("watchlists:sources.importExportFailedCsv", "Export failed CSV")}
            </Button>
            <Button onClick={handleExportFailedJson}>
              {t("watchlists:sources.importExportFailedJson", "Export failed JSON")}
            </Button>
          </Space>
        </div>
        {isConstrained ? (
          <div
            className="space-y-2"
            data-testid="opml-failed-constrained-list"
          >
            {failedItems.map((item, idx) => (
              <div
                key={`${item.url}-${idx}`}
                className="rounded-md border border-border bg-surface p-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">
                      {item.name || item.url}
                    </div>
                    <div className="break-all text-xs text-text-muted">
                      {item.url}
                    </div>
                    {item.error ? (
                      <div className="mt-1 text-xs text-danger">
                        {item.error}
                      </div>
                    ) : null}
                  </div>
                  <Tag color={isRetryableFailure(item.reasonCode) ? "orange" : "default"}>
                    {item.reasonCode}
                  </Tag>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <Table
            dataSource={failedItems}
            rowKey={(item, idx) => `${item.url}-${idx}`}
            pagination={false}
            size="small"
            columns={[
              {
                title: t("watchlists:sources.columns.name", "Name"),
                dataIndex: "name",
                key: "name",
                render: (name: string | null, record) => name || record.url
              },
              {
                title: t("watchlists:sources.columns.url", "URL"),
                dataIndex: "url",
                key: "url",
                ellipsis: true
              },
              {
                title: t("watchlists:sources.columns.status", "Status"),
                dataIndex: "status",
                key: "status"
              },
              {
                title: t("watchlists:sources.importReasonCode", "Reason code"),
                dataIndex: "reasonCode",
                key: "reasonCode",
                render: (reasonCode: ImportFailureReasonCode) => (
                  <Tag color={isRetryableFailure(reasonCode) ? "orange" : "default"}>
                    {reasonCode}
                  </Tag>
                )
              },
              {
                title: t("watchlists:sources.columns.error", "Error"),
                dataIndex: "error",
                key: "error",
                ellipsis: true
              }
            ]}
          />
        )}
      </div>
    )
  }

  return (
    <Modal
      title={t("watchlists:sources.importTitle", "Import OPML")}
      open={open}
      onCancel={onClose}
      footer={(
        <Space
          wrap
          className={isConstrained ? "w-full justify-end" : undefined}
          data-testid="sources-bulk-import-footer"
        >
          <Button onClick={onClose}>
            {t("common:close", "Close")}
          </Button>
          <Button
            type="primary"
            onClick={handleCommitImport}
            loading={importing}
            disabled={!canCommitImport}
          >
            {t(
              "watchlists:sources.importConfirm",
              "Import {{count}} feed{{plural}}",
              {
                count: preflight?.ready ?? 0,
                plural: (preflight?.ready ?? 0) === 1 ? "" : "s"
              }
            )}
          </Button>
        </Space>
      )}
      data-testid="sources-bulk-import-modal"
      width={modalChrome.width}
      style={modalChrome.style}
      styles={modalChrome.styles}
    >
      <div className="space-y-4">
        <Upload.Dragger {...uploadProps} disabled={importing}>
          <div className="flex flex-col items-center justify-center gap-2 py-6 text-text-muted">
            <UploadCloud className="h-6 w-6" />
            <div className="text-sm font-medium">
              {t("watchlists:sources.importDrop", "Drop OPML file here or click to upload")}
            </div>
            <div className="flex items-center gap-1 text-xs">
              <span>{t("watchlists:sources.importHint", "Supports standard OPML exports")}</span>
              <WatchlistsHelpTooltip topic="opml" />
            </div>
            {selectedFile && (
              <div className="text-xs text-text-subtle">
                {t("watchlists:sources.importSelectedFile", "Selected: {{file}}", {
                  file: selectedFile.name
                })}
              </div>
            )}
          </div>
        </Upload.Dragger>

        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <div className="text-xs font-medium text-text-muted mb-1">
              {t("watchlists:sources.importGroup", "Assign Group")}
            </div>
            <Select
              value={selectedGroupId ?? undefined}
              onChange={(value) => setSelectedGroupId(value ?? null)}
              allowClear
              placeholder={t("watchlists:sources.importGroupPlaceholder", "None")}
              options={groups.map((group) => ({
                label: group.name,
                value: group.id
              }))}
              className="w-full"
            />
          </div>
          <div>
            <div className="text-xs font-medium text-text-muted mb-1">
              {t("watchlists:sources.importTags", "Apply Tags")}
            </div>
            <Select
              mode="multiple"
              value={selectedTags}
              onChange={setSelectedTags}
              placeholder={t("watchlists:sources.importTagsPlaceholder", "Select tags")}
              options={tags.map((tag) => ({ label: tag.name, value: tag.name }))}
              className="w-full"
            />
          </div>
          <div>
            <div className="text-xs font-medium text-text-muted mb-1">
              {t("watchlists:sources.importActive", "Set Active")}
            </div>
            <Switch checked={active} onChange={setActive} />
          </div>
        </div>

        {existingUrlsLoading && (
          <Alert
            type="info"
            showIcon
            title={t(
              "watchlists:sources.importPreviewLoadingExisting",
              "Loading existing feeds for duplicate checks..."
            )}
          />
        )}

        {preflight && (
          <Alert
            type={preflight.parseError ? "error" : preflight.ready > 0 ? "info" : "warning"}
            showIcon
            title={t("watchlists:sources.importPreviewSummaryTitle", "Preflight Summary")}
            description={preflight.parseError
              ? t(
                  "watchlists:sources.importPreviewParseError",
                  "Could not parse this OPML file. Verify it contains outline nodes with xmlUrl attributes."
                )
              : t(
                  "watchlists:sources.importPreviewSummary",
                  "{{ready}} ready, {{duplicateExisting}} existing duplicates, {{duplicateFile}} file duplicates, {{invalidUrl}} invalid URLs, {{missingUrl}} missing URLs.",
                  {
                    ready: preflight.ready,
                    duplicateExisting: preflight.duplicateExisting,
                    duplicateFile: preflight.duplicateFile,
                    invalidUrl: preflight.invalidUrl,
                    missingUrl: preflight.missingUrl
                  }
                )}
          />
        )}

        {renderPreflightPreview()}

        {summary && (
          <Alert
            type={summary.errors > 0 ? "warning" : "success"}
            showIcon
            title={t("watchlists:sources.importSummary", "Import Summary")}
            description={t(
              "watchlists:sources.importSummaryDesc",
              "{{created}} created, {{skipped}} skipped, {{errors}} errors",
              summary
            )}
          />
        )}

        {renderFailedItems()}
      </div>
    </Modal>
  )
}
