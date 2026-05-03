import React from "react"
import { useQuery } from "@tanstack/react-query"
import { Alert, Button, Form, Input, Select, Switch, Typography } from "antd"
import { useTranslation } from "react-i18next"

import { useAntdMessage } from "@/hooks/useAntdMessage"
import { exportFlashcards, exportFlashcardsFile, listFlashcards } from "@/services/flashcards"

import { useDecksQuery } from "../../hooks"
import { type TransferActionReporterProps } from "./shared"

const { Text } = Typography

const toJsonExportText = (payload: unknown): string => {
  if (typeof payload === "string") {
    return payload
  }
  return JSON.stringify(payload, null, 2) ?? ""
}

export const ExportPanel: React.FC<TransferActionReporterProps> = ({ onTransferAction }) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const decksQuery = useDecksQuery()
  const [exportDeckId, setExportDeckId] = React.useState<number | null>(null)
  const [exportFormat, setExportFormat] = React.useState<"csv" | "apkg" | "json">("csv")
  const [exportTag, setExportTag] = React.useState("")
  const [exportQueryText, setExportQueryText] = React.useState("")
  const [exportIncludeReverse, setExportIncludeReverse] = React.useState(false)
  const [exportDelimiter, setExportDelimiter] = React.useState<string>("\t")
  const [exportIncludeHeader, setExportIncludeHeader] = React.useState(false)
  const [exportExtendedHeader, setExportExtendedHeader] = React.useState(false)
  const [isExporting, setIsExporting] = React.useState(false)

  const normalizedExportTag = exportTag.trim()
  const normalizedExportTagLower = normalizedExportTag.toLowerCase()
  const normalizedExportQuery = exportQueryText.trim()

  const exportPreviewCountQuery = useQuery({
    queryKey: [
      "flashcards:export-preview-count",
      exportDeckId ?? null,
      normalizedExportTagLower,
      normalizedExportQuery
    ],
    queryFn: async () => {
      const response = await listFlashcards({
        deck_id: exportDeckId ?? undefined,
        tag: normalizedExportTagLower || undefined,
        q: normalizedExportQuery || undefined,
        due_status: "all",
        limit: 1,
        offset: 0
      })
      return response.total ?? response.count ?? 0
    }
  })

  const selectedDeckLabel = React.useMemo(() => {
    if (exportDeckId == null) {
      return t("option:flashcards.allDecks", {
        defaultValue: "All decks"
      })
    }
    return (
      (decksQuery.data || []).find((deck) => deck.id === exportDeckId)?.name ||
      `${t("option:flashcards.deck", { defaultValue: "Deck" })} ${exportDeckId}`
    )
  }, [decksQuery.data, exportDeckId, t])

  const handleExport = async () => {
    setIsExporting(true)
    try {
      const exportParams = {
        deck_id: exportDeckId ?? undefined,
        tag: normalizedExportTagLower || undefined,
        q: normalizedExportQuery || undefined,
        include_reverse: exportIncludeReverse || undefined
      }

      let blob: Blob
      if (exportFormat === "apkg") {
        blob = await exportFlashcardsFile({
          ...exportParams,
          format: "apkg"
        })
      } else if (exportFormat === "json") {
        const payload = await exportFlashcards({
          ...exportParams,
          format: "json"
        })
        blob = new Blob([toJsonExportText(payload)], {
          type: "application/json;charset=utf-8"
        })
      } else {
        const text = await exportFlashcards({
          ...exportParams,
          format: "csv",
          delimiter: exportDelimiter,
          include_header: exportIncludeHeader,
          extended_header: exportExtendedHeader
        })
        blob = new Blob([text], {
          type:
            exportDelimiter === "\t"
              ? "text/tab-separated-values;charset=utf-8"
              : "text/csv;charset=utf-8"
        })
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      if (exportFormat === "apkg") {
        a.download = "flashcards.apkg"
      } else if (exportFormat === "json") {
        a.download = "flashcards.json"
      } else {
        a.download = exportDelimiter === "\t" ? "flashcards.tsv" : "flashcards.csv"
      }
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      const successCopy = t("option:flashcards.exportSuccess", {
        defaultValue: "Export ready: {{fileName}}",
        fileName:
          exportFormat === "apkg"
            ? "flashcards.apkg"
            : exportFormat === "json"
              ? "flashcards.json"
              : exportDelimiter === "\t"
                ? "flashcards.tsv"
                : "flashcards.csv"
      })
      message.success(successCopy)
      onTransferAction?.({
        area: "export",
        status: "success",
        message: successCopy
      })
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Export failed"
      message.error(errorMessage)
      onTransferAction?.({
        area: "export",
        status: "error",
        message: errorMessage
      })
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div>
        <Text type="secondary">
          {t("option:flashcards.exportHelp", {
            defaultValue:
              "Export filtered flashcards to delimited text (CSV/TSV), JSON, or Anki-compatible APKG format."
          })}
        </Text>
      </div>
      <Form.Item
        label={t("option:flashcards.deck", { defaultValue: "Deck" })}
        className="!mb-2"
      >
        <Select
          placeholder={t("option:flashcards.allDecks", {
            defaultValue: "All decks"
          })}
          allowClear
          loading={decksQuery.isLoading}
          value={exportDeckId ?? undefined}
          onChange={setExportDeckId}
          data-testid="flashcards-export-deck"
          options={(decksQuery.data || []).map((d) => ({
            label: d.name,
            value: d.id
          }))}
        />
      </Form.Item>
      <Form.Item
        label={t("option:flashcards.exportFormat", { defaultValue: "Format" })}
        className="!mb-2"
      >
        <Select
          value={exportFormat}
          onChange={setExportFormat}
          data-testid="flashcards-export-format"
          options={[
            {
              label: t("option:flashcards.exportFormatDelimited", {
                defaultValue: "Delimited (CSV/TSV)"
              }),
              value: "csv"
            },
            {
              label: t("option:flashcards.exportFormatJson", {
                defaultValue: "JSON"
              }),
              value: "json"
            },
            { label: "APKG (Anki)", value: "apkg" }
          ]}
        />
      </Form.Item>
      <Form.Item
        label={t("option:flashcards.exportTagFilter", { defaultValue: "Tag filter" })}
        className="!mb-2"
      >
        <Input
          value={exportTag}
          onChange={(event) => setExportTag(event.target.value)}
          placeholder={t("option:flashcards.exportTagFilterPlaceholder", {
            defaultValue: "Optional single tag"
          })}
          data-testid="flashcards-export-tag"
        />
      </Form.Item>
      <Form.Item
        label={t("option:flashcards.exportQueryFilter", { defaultValue: "Text filter" })}
        className="!mb-2"
      >
        <Input
          value={exportQueryText}
          onChange={(event) => setExportQueryText(event.target.value)}
          placeholder={t("option:flashcards.exportQueryFilterPlaceholder", {
            defaultValue: "Optional search query"
          })}
          data-testid="flashcards-export-query"
        />
      </Form.Item>
      <div className="flex items-center justify-between gap-3">
        <Text>{t("option:flashcards.exportIncludeReverse", { defaultValue: "Include reverse cards" })}</Text>
        <Switch
          checked={exportIncludeReverse}
          onChange={setExportIncludeReverse}
          data-testid="flashcards-export-include-reverse"
        />
      </div>
      {exportFormat === "csv" && (
        <>
          <Form.Item
            label={t("option:flashcards.exportDelimiter", { defaultValue: "Delimiter" })}
            className="!mb-2"
          >
            <Select
              value={exportDelimiter}
              onChange={setExportDelimiter}
              data-testid="flashcards-export-delimiter"
              options={[
                {
                  label: t("option:flashcards.tab", { defaultValue: "Tab" }),
                  value: "\t"
                },
                {
                  label: t("option:flashcards.comma", { defaultValue: ", (Comma)" }),
                  value: ","
                },
                {
                  label: t("option:flashcards.semicolon", {
                    defaultValue: "; (Semicolon)"
                  }),
                  value: ";"
                },
                {
                  label: t("option:flashcards.pipe", { defaultValue: "| (Pipe)" }),
                  value: "|"
                }
              ]}
            />
          </Form.Item>
          <div className="flex items-center justify-between gap-3">
            <Text>{t("option:flashcards.exportIncludeHeader", { defaultValue: "Include header row" })}</Text>
            <Switch
              checked={exportIncludeHeader}
              onChange={setExportIncludeHeader}
              data-testid="flashcards-export-include-header"
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <Text>{t("option:flashcards.exportExtendedHeader", { defaultValue: "Use extended header columns" })}</Text>
            <Switch
              checked={exportExtendedHeader}
              onChange={setExportExtendedHeader}
              data-testid="flashcards-export-extended-header"
            />
          </div>
        </>
      )}
      <Alert
        type="info"
        showIcon
        data-testid="flashcards-export-preview"
        title={t("option:flashcards.exportPreviewTitle", {
          defaultValue: "Export preview"
        })}
        description={t("option:flashcards.exportPreviewDescription", {
          defaultValue:
            "{{count}} cards from {{deck}}. Tag filter: {{tag}}. Query filter: {{query}}.",
          count: exportPreviewCountQuery.data ?? 0,
          deck: selectedDeckLabel,
          tag:
            normalizedExportTag ||
            t("option:flashcards.noneLabel", { defaultValue: "none" }),
          query:
            normalizedExportQuery ||
            t("option:flashcards.noneLabel", { defaultValue: "none" })
        })}
      />
      <Button
        type="primary"
        onClick={handleExport}
        loading={isExporting}
        disabled={exportPreviewCountQuery.isLoading}
        data-testid="flashcards-export-button"
      >
        {t("option:flashcards.exportButton", { defaultValue: "Export" })}
      </Button>
    </div>
  )
}
