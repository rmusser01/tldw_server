import { useMemo } from "react"
import { Tooltip, message } from "antd"
import {
  CornerUpLeft,
  CopyIcon,
  DownloadIcon,
  Pin,
  PinOff,
  X,
  PlayCircle,
  Table2
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { Highlight, themes } from "prism-react-renderer"
import { browser } from "wxt/browser"
import { useArtifactsStore } from "@/store/artifacts"
import { useStoreMessageOption } from "@/store/option"
import { Mermaid } from "@/components/Common/Mermaid"
import type { DataTable, DataTableColumn, DataTableSource } from "@/types/data-tables"
import { queueDataTablesPrefill } from "@/utils/data-tables-prefill"

const normalizeLanguage = (language?: string): string => {
  const lang = (language || "").toLowerCase()
  if (lang === "js" || lang === "jsx") return "javascript"
  if (lang === "ts" || lang === "tsx") return "typescript"
  if (lang === "sh" || lang === "bash") return "bash"
  if (lang === "py") return "python"
  if (lang === "md" || lang === "markdown") return "markdown"
  if (lang === "yml") return "yaml"
  if (!lang) return "plaintext"
  return lang
}

const openExtensionUrl = (
  path: `/options.html${string}` | `/sidepanel.html${string}`
) => {
  try {
    if (browser?.runtime?.getURL) {
      const url = browser.runtime.getURL(path)
      if (browser.tabs?.create) {
        browser.tabs.create({ url })
      } else {
        window.open(url, "_blank")
      }
      return
    }
  } catch (err) {
    console.debug("[ArtifactsPanel] openExtensionUrl browser API unavailable:", err)
  }

  try {
    if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
      const url = chrome.runtime.getURL(path)
      window.open(url, "_blank")
      return
    }
    if (
      typeof chrome !== "undefined" &&
      chrome.runtime?.openOptionsPage &&
      path.includes("/options.html")
    ) {
      chrome.runtime.openOptionsPage()
      return
    }
  } catch (err) {
    console.debug("[ArtifactsPanel] openExtensionUrl chrome API unavailable:", err)
  }

  window.open(path, "_blank")
}

const buildUniqueColumnNames = (headers: string[]): string[] => {
  const seen = new Set<string>()
  return headers.map((raw, index) => {
    const base = raw?.trim() || `Column ${index + 1}`
    let name = base
    let count = 1
    while (seen.has(name)) {
      count += 1
      name = `${base} (${count})`
    }
    seen.add(name)
    return name
  })
}

const buildDataTableFromArtifact = (
  title: string,
  table: { headers: string[]; rows: string[][] },
  prompt: string,
  source?: DataTableSource
): DataTable => {
  const columnNames = buildUniqueColumnNames(table.headers)
  const columns: DataTableColumn[] = columnNames.map((name, index) => ({
    id: `col-${index + 1}`,
    name,
    type: "text"
  }))
  const rows = table.rows.map((row) => {
    const entry: Record<string, string> = {}
    columnNames.forEach((name, idx) => {
      entry[name] = row?.[idx] ?? ""
    })
    return entry
  })
  const timestamp = new Date().toISOString()
  return {
    id: `artifact-${Date.now()}`,
    name: title || "Imported Table",
    description: undefined,
    prompt,
    columns,
    rows,
    sources: source ? [source] : [],
    created_at: timestamp,
    updated_at: timestamp,
    row_count: rows.length
  }
}

export const ArtifactsPanel = () => {
  const { t } = useTranslation(["common", "dataTables"])
  const [codeTheme] = useStorage("codeTheme", "auto")
  const { active, isOpen, isPinned, closeArtifact, setPinned } =
    useArtifactsStore()
  const { serverChatId, serverChatTitle } = useStoreMessageOption((state) => ({
    serverChatId: state.serverChatId,
    serverChatTitle: state.serverChatTitle
  }))

  const normalizedLanguage = useMemo(
    () => normalizeLanguage(active?.language),
    [active?.language]
  )

  const resolveTheme = (key: string) => {
    if (key === "auto") {
      let isDark = false
      try {
        if (typeof document !== "undefined") {
          const root = document.documentElement
          if (root.classList.contains("dark")) {
            isDark = true
          } else if (root.classList.contains("light")) {
            isDark = false
          } else if (typeof window !== "undefined") {
            isDark = window.matchMedia("(prefers-color-scheme: dark)").matches
          }
        }
      } catch {
        isDark = false
      }
      return isDark ? themes.dracula : themes.github
    }
    switch (key) {
      case "github":
        return themes.github
      case "nightOwl":
        return themes.nightOwl
      case "nightOwlLight":
        return themes.nightOwlLight
      case "vsDark":
        return themes.vsDark
      case "dracula":
      default:
        return themes.dracula
    }
  }

  if (!isOpen || !active) {
    return null
  }

  const isCodeArtifact = active.kind === "code" || active.kind === "diagram"
  const highlightLanguage =
    active.kind === "diagram" ? "markdown" : normalizedLanguage

  const handleCopy = () => {
    navigator.clipboard.writeText(active.content)
  }

  const handleDownload = () => {
    const blob = new Blob([active.content], { type: "text/plain" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `artifact_${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.${normalizedLanguage || "txt"}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  const handleJumpToSource = () => {
    const focusArtifactsTrigger = () => {
      window.setTimeout(() => {
        window.dispatchEvent(new CustomEvent("tldw:focus-artifacts-trigger"))
      }, 0)
    }
    if (typeof document !== "undefined" && active?.id) {
      const origin = document.getElementById(`artifact-origin-${active.id}`)
      if (origin && typeof origin.scrollIntoView === "function") {
        origin.scrollIntoView({ behavior: "smooth", block: "center" })
        closeArtifact()
        focusArtifactsTrigger()
        return
      }
    }
    closeArtifact()
    window.dispatchEvent(new CustomEvent("tldw:scroll-to-latest"))
    focusArtifactsTrigger()
  }

  const handleSaveToDataTables = async () => {
    if (!active?.table) {
      message.error(t("dataTables:noTableToSave", "No table to save"))
      return
    }
    const promptText = t(
      "dataTables:importedPrompt",
      "Imported from table artifact"
    )
    const source =
      serverChatId != null
        ? {
            type: "chat" as const,
            id: String(serverChatId),
            title: serverChatTitle || active.title || `Chat ${serverChatId}`
          }
        : undefined
    const table = buildDataTableFromArtifact(
      active.title,
      active.table,
      promptText,
      source
    )
    await queueDataTablesPrefill({ kind: "artifact", table, source })
    openExtensionUrl("/options.html#/data-tables")
  }

  return (
    <aside
      data-testid="artifacts-panel"
      className="flex h-full min-w-[280px] flex-col border-l border-border bg-surface/95 backdrop-blur">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <div className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("artifactsTitle", "Artifact")}
          </div>
          <div className="truncate text-sm font-medium text-text">
            {active.title}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Tooltip
            title={
              isPinned
                ? t("artifactsUnpin", "Unpin")
                : t("artifactsPin", "Pin")
            }>
            <button
              type="button"
              onClick={() => setPinned(!isPinned)}
              className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              title={
                isPinned
                  ? t("artifactsUnpin", "Unpin")
                  : t("artifactsPin", "Pin")
              }
            >
              {isPinned ? (
                <PinOff className="h-4 w-4" />
              ) : (
                <Pin className="h-4 w-4" />
              )}
            </button>
          </Tooltip>
          <Tooltip title={t("artifactsClose", "Close")}>
            <button
              type="button"
              data-testid="artifacts-panel-close"
              aria-label={t("artifactsClose", "Close") as string}
              onClick={() => {
                closeArtifact()
                window.setTimeout(() => {
                  window.dispatchEvent(
                    new CustomEvent("tldw:focus-artifacts-trigger")
                  )
                }, 0)
              }}
              className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              title={t("artifactsClose", "Close")}
            >
              <X className="h-4 w-4" />
            </button>
          </Tooltip>
        </div>
      </div>
      <div className="flex-1 overflow-auto px-4 py-3">
        {active.kind === "diagram" ? (
          <Mermaid
            code={active.content}
            className="rounded-xl border border-border bg-surface2/60 px-4 py-3"
          />
        ) : isCodeArtifact ? (
          <Highlight
            code={active.content}
            language={highlightLanguage as any}
            theme={resolveTheme(codeTheme || "dracula")}>
            {({
              className: highlightClassName,
              style,
              tokens,
              getLineProps,
              getTokenProps
            }) => (
              <pre
                className={`${highlightClassName} m-0 rounded-xl bg-surface2/80 px-4 py-3 text-[0.85rem]`}
                style={{
                  ...style,
                  fontFamily: "var(--font-mono)"
                }}>
                {tokens.map((line, i) => (
                  <div
                    key={i}
                    {...getLineProps({ line, key: i })}
                    className="table w-full">
                    <span className="table-cell select-none pr-3 text-right text-xs text-text-subtle">
                      {i + 1}
                    </span>
                    <span className="table-cell whitespace-pre-wrap">
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token, key })} />
                      ))}
                    </span>
                  </div>
                ))}
              </pre>
            )}
          </Highlight>
        ) : active.kind === "table" && active.table ? (
          <div className="overflow-auto rounded-xl border border-border bg-surface2/60">
            <table className="w-full border-collapse text-xs text-text">
              {active.table.headers.length > 0 && (
                <thead className="bg-surface2">
                  <tr>
                    {active.table.headers.map((header, index) => (
                      <th
                        key={`${header}-${index}`}
                        className="whitespace-nowrap border-b border-border px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wide text-text-subtle">
                        {header || "-"}
                      </th>
                    ))}
                  </tr>
                </thead>
              )}
              <tbody>
                {active.table.rows.map((row, rowIndex) => (
                  <tr key={`row-${rowIndex}`} className="odd:bg-surface/60">
                    {row.map((cell, cellIndex) => (
                      <td
                        key={`cell-${rowIndex}-${cellIndex}`}
                        className="whitespace-nowrap border-b border-border px-3 py-2 text-xs text-text">
                        {cell || "-"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <pre className="whitespace-pre-wrap rounded-xl bg-surface2/80 px-4 py-3 text-xs text-text">
            {active.content}
          </pre>
        )}
      </div>
      <div className="flex items-center gap-2 border-t border-border px-4 py-2">
        <Tooltip title={t("artifactsJumpToSource", "Jump to source message")}>
          <button
            type="button"
            data-testid="artifacts-jump-source"
            onClick={handleJumpToSource}
            className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-text hover:bg-surface"
            title={t("artifactsJumpToSource", "Jump to source message")}
          >
            <CornerUpLeft className="h-3 w-3" />
            <span>{t("artifactsJumpToSourceShort", "Jump to source")}</span>
          </button>
        </Tooltip>
        <Tooltip title={t("artifactsCopy", "Copy")}>
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-text hover:bg-surface"
            title={t("artifactsCopy", "Copy")}
          >
            <CopyIcon className="h-3 w-3" />
            <span>{t("artifactsCopy", "Copy")}</span>
          </button>
        </Tooltip>
        <Tooltip title={t("artifactsDownload", "Download")}>
          <button
            type="button"
            onClick={handleDownload}
            className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-text hover:bg-surface"
            title={t("artifactsDownload", "Download")}
          >
            <DownloadIcon className="h-3 w-3" />
            <span>{t("artifactsDownload", "Download")}</span>
          </button>
        </Tooltip>
        {active.kind === "table" && active.table && (
          <Tooltip title={t("dataTables:saveFromArtifact", "Save to Data Tables")}>
            <button
              type="button"
              onClick={handleSaveToDataTables}
              className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-text hover:bg-surface"
              title={t("dataTables:saveFromArtifact", "Save to Data Tables")}
            >
              <Table2 className="h-3 w-3" />
              <span>{t("dataTables:saveFromArtifact", "Save to Data Tables")}</span>
            </button>
          </Tooltip>
        )}
        <button
          type="button"
          disabled
          className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted opacity-70"
          title={t("artifactsRun", "Run (N/A)")}
        >
          <PlayCircle className="h-3 w-3" />
          <span>{t("artifactsRun", "Run (N/A)")}</span>
        </button>
      </div>
    </aside>
  )
}
