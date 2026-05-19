/**
 * DiffViewer - Display unified diffs with hunk selection
 */

import { FC, useState, useMemo, useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  FileText,
  Plus,
  Minus,
  ChevronDown,
  ChevronRight,
  Check,
  X,
  Copy,
  CheckCheck
} from "lucide-react"
import { message } from "antd"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import { Badge, getBadgeVariantForDesignSystemSeverity } from "@/components/ui/primitives"

export interface DiffHunk {
  id: string
  oldStart: number
  oldCount: number
  newStart: number
  newCount: number
  lines: DiffLine[]
}

export interface DiffLine {
  type: "context" | "add" | "remove" | "header"
  content: string
  oldLineNum?: number
  newLineNum?: number
}

export interface FileDiff {
  id: string
  oldPath: string
  newPath: string
  hunks: DiffHunk[]
  isNew?: boolean
  isDeleted?: boolean
  isRenamed?: boolean
}

interface DiffViewerProps {
  diffs: FileDiff[]
  selectedHunks?: Set<string>
  onHunkSelectionChange?: (hunkIds: Set<string>) => void
  // Currently only unified view is implemented
  viewMode?: "unified"
  className?: string
  collapsible?: boolean
  showLineNumbers?: boolean
}

/**
 * Parse a unified diff string into structured FileDiff objects
 */
export function parseDiff(diffText: string): FileDiff[] {
  const files: FileDiff[] = []
  const lines = diffText.split("\n")
  let currentFile: FileDiff | null = null
  let currentHunk: DiffHunk | null = null
  let oldLineNum = 0
  let newLineNum = 0

  const finalizeFile = (file: FileDiff, hunk: DiffHunk | null) => {
    if (hunk) {
      file.hunks.push(hunk)
    }
    if (!file.isNew && !file.isDeleted && file.oldPath && file.newPath && file.oldPath !== file.newPath) {
      file.isRenamed = true
    }
    files.push(file)
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // File header: diff --git a/path b/path
    if (line.startsWith("diff --git")) {
      if (currentFile) {
        finalizeFile(currentFile, currentHunk)
      }
      currentFile = {
        id: `file-${files.length}`,
        oldPath: "",
        newPath: "",
        hunks: []
      }
      currentHunk = null
      continue
    }

    // Old file path: --- a/path or --- /dev/null
    if (line.startsWith("--- ")) {
      if (currentFile) {
        const path = line.slice(4)
        currentFile.oldPath = path.startsWith("a/") ? path.slice(2) : path
        if (path === "/dev/null") {
          currentFile.isNew = true
        }
      }
      continue
    }

    // New file path: +++ b/path or +++ /dev/null
    if (line.startsWith("+++ ")) {
      if (currentFile) {
        const path = line.slice(4)
        currentFile.newPath = path.startsWith("b/") ? path.slice(2) : path
        if (path === "/dev/null") {
          currentFile.isDeleted = true
        }
      }
      continue
    }

    // Rename metadata: "rename from"/"rename to"
    if (line.startsWith("rename from ")) {
      if (currentFile) {
        currentFile.isRenamed = true
        const path = line.slice("rename from ".length)
        currentFile.oldPath = path.startsWith("a/") ? path.slice(2) : path
      }
      continue
    }
    if (line.startsWith("rename to ")) {
      if (currentFile) {
        currentFile.isRenamed = true
        const path = line.slice("rename to ".length)
        currentFile.newPath = path.startsWith("b/") ? path.slice(2) : path
      }
      continue
    }

    // Hunk header: @@ -oldStart,oldCount +newStart,newCount @@
    const hunkMatch = line.match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$/)
    if (hunkMatch && currentFile) {
      if (currentHunk) {
        currentFile.hunks.push(currentHunk)
      }
      oldLineNum = parseInt(hunkMatch[1], 10)
      newLineNum = parseInt(hunkMatch[3], 10)
      currentHunk = {
        id: `${currentFile.id}-hunk-${currentFile.hunks.length}`,
        oldStart: oldLineNum,
        oldCount: parseInt(hunkMatch[2] || "1", 10),
        newStart: newLineNum,
        newCount: parseInt(hunkMatch[4] || "1", 10),
        lines: [{
          type: "header",
          content: line
        }]
      }
      continue
    }

    // Diff content lines
    if (currentHunk) {
      if (line.startsWith("+")) {
        currentHunk.lines.push({
          type: "add",
          content: line.slice(1),
          newLineNum: newLineNum++
        })
      } else if (line.startsWith("-")) {
        currentHunk.lines.push({
          type: "remove",
          content: line.slice(1),
          oldLineNum: oldLineNum++
        })
      } else if (line.startsWith(" ")) {
        currentHunk.lines.push({
          type: "context",
          content: line.slice(1) || "",
          oldLineNum: oldLineNum++,
          newLineNum: newLineNum++
        })
      }
    }
  }

  // Push last file and hunk
  if (currentFile) {
    finalizeFile(currentFile, currentHunk)
  }

  return files
}

/**
 * Get display name for a file diff
 */
function getFilePath(diff: FileDiff): string {
  if (diff.isNew) return diff.newPath
  if (diff.isDeleted) return diff.oldPath
  if (diff.isRenamed) return `${diff.oldPath} → ${diff.newPath}`
  return diff.newPath || diff.oldPath
}

/**
 * Get file status badge
 */
const FILE_STATUS_CONFIG = {
  new: {
    label: "NEW",
    srLabel: "New file",
    stateKey: "ready"
  },
  deleted: {
    label: "DEL",
    srLabel: "Deleted file",
    stateKey: "error"
  },
  renamed: {
    label: "RENAME",
    srLabel: "Renamed file",
    stateKey: "empty"
  }
} satisfies Record<string, {
  label: string
  srLabel: string
  stateKey: DesignSystemStateKey
}>

const FileStatusBadge: FC<{ diff: FileDiff }> = ({ diff }) => {
  const config = diff.isNew
    ? FILE_STATUS_CONFIG.new
    : diff.isDeleted
      ? FILE_STATUS_CONFIG.deleted
      : diff.isRenamed
        ? FILE_STATUS_CONFIG.renamed
        : null

  if (!config) {
    return null
  }

  const state = getDesignSystemState(config.stateKey)

  return (
    <Badge
      variant={getBadgeVariantForDesignSystemSeverity(state.severity)}
      size="md"
      pill={false}
      srLabel={config.srLabel}
    >
      {config.label}
    </Badge>
  )
}

/**
 * Line number component
 */
const LineNumber: FC<{ num?: number; className?: string }> = ({ num, className = "" }) => (
  <span className={`select-none text-text-subtle min-w-10 text-right pr-2 ${className}`}>
    {num ?? ""}
  </span>
)

export const DiffViewer: FC<DiffViewerProps> = ({
  diffs,
  selectedHunks,
  onHunkSelectionChange,
  viewMode = "unified",
  className = "",
  collapsible = true,
  showLineNumbers = true
}) => {
  const { t } = useTranslation("common")
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(
    () => new Set(diffs.map(d => d.id))
  )
  const [copiedHunk, setCopiedHunk] = useState<string | null>(null)

  // Track selected hunks internally if not controlled
  const [internalSelected, setInternalSelected] = useState<Set<string>>(
    () => new Set(diffs.flatMap(d => d.hunks.map(h => h.id)))
  )
  const selected = selectedHunks ?? internalSelected
  const setSelected = onHunkSelectionChange ?? setInternalSelected

  useEffect(() => {
    setExpandedFiles((prev) => {
      let changed = false
      const next = new Set(prev)
      for (const diff of diffs) {
        if (!next.has(diff.id)) {
          next.add(diff.id)
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [diffs])

  useEffect(() => {
    if (selectedHunks) {
      return
    }
    setInternalSelected((prev) => {
      let changed = false
      const next = new Set(prev)
      for (const diff of diffs) {
        for (const hunk of diff.hunks) {
          if (!next.has(hunk.id)) {
            next.add(hunk.id)
            changed = true
          }
        }
      }
      return changed ? next : prev
    })
  }, [diffs, selectedHunks])

  // Count changes
  const stats = useMemo(() => {
    let additions = 0
    let deletions = 0
    for (const diff of diffs) {
      for (const hunk of diff.hunks) {
        for (const line of hunk.lines) {
          if (line.type === "add") additions++
          if (line.type === "remove") deletions++
        }
      }
    }
    return { additions, deletions }
  }, [diffs])

  const toggleFile = (fileId: string) => {
    const next = new Set(expandedFiles)
    if (next.has(fileId)) {
      next.delete(fileId)
    } else {
      next.add(fileId)
    }
    setExpandedFiles(next)
  }

  const toggleHunk = (hunkId: string) => {
    const next = new Set(selected)
    if (next.has(hunkId)) {
      next.delete(hunkId)
    } else {
      next.add(hunkId)
    }
    setSelected(next)
  }

  const selectAllHunks = () => {
    setSelected(new Set(diffs.flatMap(d => d.hunks.map(h => h.id))))
  }

  const deselectAllHunks = () => {
    setSelected(new Set())
  }

  const copyHunk = async (hunk: DiffHunk) => {
    const text = hunk.lines
      .filter(l => l.type !== "header")
      .map(l => {
        if (l.type === "add") return `+${l.content}`
        if (l.type === "remove") return `-${l.content}`
        return ` ${l.content}`
      })
      .join("\n")

    try {
      await navigator.clipboard.writeText(text)
      setCopiedHunk(hunk.id)
      message.success(t("copiedToClipboard", "Copied to clipboard"))
      setTimeout(() => setCopiedHunk(null), 2000)
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Failed to copy hunk to clipboard:", err)
      message.error(t("copyFailed", "Failed to copy"))
    }
  }

  if (diffs.length === 0) {
    return (
      <div className={`flex h-32 items-center justify-center text-text-subtle ${className}`}>
        <span className="text-sm">{t("noDiffs", "No changes to display")}</span>
      </div>
    )
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Summary header */}
      <div className="flex items-center justify-between rounded-lg bg-surface2 px-3 py-2">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium">
            {diffs.length}{" "}
            {diffs.length === 1
              ? t("fileChanged", "file changed")
              : t("filesChanged", "files changed")}
          </span>
          <span className="flex items-center gap-1 text-sm text-success">
            <Plus className="size-3" />
            {stats.additions}
          </span>
          <span className="flex items-center gap-1 text-sm text-danger">
            <Minus className="size-3" />
            {stats.deletions}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={selectAllHunks}
            className="rounded px-2 py-1 text-xs hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            aria-label={t("selectAllHunks", "Select all diff hunks")}
          >
            {t("selectAll", "Select All")}
          </button>
          <button
            onClick={deselectAllHunks}
            className="rounded px-2 py-1 text-xs hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            aria-label={t("deselectAllHunks", "Deselect all diff hunks")}
          >
            {t("deselectAll", "Deselect All")}
          </button>
        </div>
      </div>

      {/* File diffs */}
      {diffs.map((diff) => {
        const isExpanded = expandedFiles.has(diff.id)
        const fileSelected = diff.hunks.every(h => selected.has(h.id))
        const filePartial = diff.hunks.some(h => selected.has(h.id)) && !fileSelected

        return (
          <div
            key={diff.id}
            className="overflow-hidden rounded-lg border border-border"
          >
            {/* File header */}
            <button
              onClick={() => collapsible && toggleFile(diff.id)}
              disabled={!collapsible}
              className={`w-full flex items-center gap-2 bg-surface2 px-3 py-2 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus ${
                collapsible ? "hover:bg-surface" : "cursor-default"
              }`}
              aria-expanded={collapsible ? isExpanded : undefined}
              aria-label={`${getFilePath(diff)} - ${isExpanded ? t("collapse", "Collapse") : t("expand", "Expand")}`}
            >
              {collapsible && (
                isExpanded
                  ? <ChevronDown className="size-4 text-text-subtle" />
                  : <ChevronRight className="size-4 text-text-subtle" />
              )}

              <FileText className="size-4 text-text-subtle" />

              <span className="font-mono text-sm flex-1 truncate">
                {getFilePath(diff)}
              </span>

              <FileStatusBadge diff={diff} />

              {/* File selection indicator */}
              <div className={`flex size-4 items-center justify-center rounded border ${
                fileSelected
                  ? "border-primary bg-primary"
                  : filePartial
                    ? "border-primary bg-primary/30"
                    : "border-border-strong"
              }`}>
                {fileSelected && <Check className="size-3 text-white" />}
                {filePartial && <Minus className="size-3 text-primary" />}
              </div>
            </button>

            {/* Hunks */}
            {isExpanded && (
              <div className="divide-y divide-border">
                {diff.hunks.map((hunk) => {
                  const isSelected = selected.has(hunk.id)

                  return (
                    <div key={hunk.id} className="relative">
                      {/* Hunk header */}
                      <div className="flex items-center justify-between border-b border-border bg-primary/10 px-3 py-1">
                        <span className="font-mono text-xs text-primary">
                          @@ -{hunk.oldStart},{hunk.oldCount} +{hunk.newStart},{hunk.newCount} @@
                        </span>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => copyHunk(hunk)}
                            className="rounded p-1 hover:bg-primary/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                            title={t("copyHunk", "Copy hunk")}
                            aria-label={t("copyHunk", "Copy hunk")}
                          >
                            {copiedHunk === hunk.id ? (
                              <CheckCheck className="size-3.5 text-success" />
                            ) : (
                              <Copy className="size-3.5 text-text-subtle" />
                            )}
                          </button>
                          <button
                            onClick={() => toggleHunk(hunk.id)}
                            className={`flex size-5 items-center justify-center rounded border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-focus ${
                              isSelected
                                ? "border-primary bg-primary"
                                : "border-border-strong hover:border-primary"
                            }`}
                            title={isSelected ? t("deselectHunk", "Deselect hunk") : t("selectHunk", "Select hunk")}
                            aria-label={isSelected ? t("deselectHunk", "Deselect hunk") : t("selectHunk", "Select hunk")}
                            aria-pressed={isSelected}
                          >
                            {isSelected && <Check className="size-3 text-white" />}
                          </button>
                        </div>
                      </div>

                      {/* Diff lines */}
                      <div className={`overflow-x-auto font-mono text-sm ${!isSelected ? "opacity-50" : ""}`}>
                        {hunk.lines.filter(l => l.type !== "header").map((line, idx) => (
                          <div
                            key={idx}
                          className={`flex ${
                              line.type === "add"
                                ? "bg-success/10"
                                : line.type === "remove"
                                  ? "bg-danger/10"
                                  : ""
                            }`}
                          >
                            {showLineNumbers && (
                              <>
                                <LineNumber num={line.oldLineNum} className="border-r border-border" />
                                <LineNumber num={line.newLineNum} className="border-r border-border" />
                              </>
                            )}
                            <span className={`w-5 flex-shrink-0 select-none text-center ${
                              line.type === "add"
                                ? "text-success"
                                : line.type === "remove"
                                  ? "text-danger"
                                  : "text-text-subtle"
                            }`}>
                              {line.type === "add" ? "+" : line.type === "remove" ? "-" : " "}
                            </span>
                            <pre className="flex-1 px-2 whitespace-pre">
                              {line.content}
                            </pre>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default DiffViewer
