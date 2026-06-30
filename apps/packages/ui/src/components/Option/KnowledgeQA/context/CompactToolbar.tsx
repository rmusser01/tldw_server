import React from "react"
import { Popover } from "antd"
import { Layers, Globe, ChevronDown, Settings, FolderPlus } from "lucide-react"
import { cn } from "@/libs/utils"
import type { RagPresetName, RagSource } from "@/services/rag/unified-rag"
import { ALL_RAG_SOURCES, getRagSourceLabel } from "@/services/rag/sourceMetadata"
import { AnswerModelMenu } from "./AnswerModelMenu"
import type { KnowledgeSourceHealthState } from "../types"
import { buildSourceHealthSummary } from "../sourceHealth"

type CompactToolbarProps = {
  sources: RagSource[]
  includeMediaIds?: number[]
  includeNoteIds?: string[]
  preset: RagPresetName
  webEnabled: boolean
  webFallbackAvailable?: boolean
  onToggleWeb: () => void
  onOpenSourceSelector: () => void
  onAddSources?: () => void
  onOpenSettings: () => void
  generationProvider: string | null
  generationModel: string | null
  onGenerationProviderChange: (provider: string | null) => void
  onGenerationModelChange: (model: string | null) => void
  contextChangedSinceLastRun: boolean
  scopeChangeDetails?: string[]
  sourceHealth?: KnowledgeSourceHealthState
  onRefreshSourceHealth?: () => void
  showAddSources?: boolean
  className?: string
}

const ALL_SOURCES_THRESHOLD = ALL_RAG_SOURCES.length

function summarizeSources(sources: RagSource[]): string {
  if (!Array.isArray(sources) || sources.length === 0) return "None"
  if (sources.length === 1) return getRagSourceLabel(sources[0])
  if (sources.length >= ALL_SOURCES_THRESHOLD) return "All sources"
  return `${sources.length} selected`
}

function summarizeSpecificSources(mediaIds: number[], noteIds: string[]): string | null {
  const mediaCount = mediaIds.filter((id) => Number.isFinite(id) && id > 0).length
  const noteCount = noteIds.filter((id) => typeof id === "string" && id.trim().length > 0).length
  if (mediaCount === 0 && noteCount === 0) return null

  const parts: string[] = []
  if (mediaCount > 0) {
    parts.push(`${mediaCount} doc${mediaCount === 1 ? "" : "s"}`)
  }
  if (noteCount > 0) {
    parts.push(`${noteCount} note${noteCount === 1 ? "" : "s"}`)
  }
  return parts.join(" • ")
}

const PRESET_LABELS: Record<string, string> = {
  fast: "Fast",
  balanced: "Balanced",
  thorough: "Deep",
  custom: "Custom",
}

export function CompactToolbar({
  sources,
  includeMediaIds = [],
  includeNoteIds = [],
  preset,
  webEnabled,
  webFallbackAvailable = true,
  onToggleWeb,
  onOpenSourceSelector,
  onAddSources,
  onOpenSettings,
  generationProvider,
  generationModel,
  onGenerationProviderChange,
  onGenerationModelChange,
  contextChangedSinceLastRun,
  scopeChangeDetails = [],
  sourceHealth,
  onRefreshSourceHealth,
  showAddSources = false,
  className,
}: CompactToolbarProps) {
  const sourceSummary = summarizeSources(sources)
  const specificSourceSummary = summarizeSpecificSources(includeMediaIds, includeNoteIds)
  const sourceControlLabel = `Open source scope and saved profiles. Sources: ${sourceSummary}${
    specificSourceSummary ? `. Specific: ${specificSourceSummary}` : ""
  }`

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      {showAddSources ? (
        <button
          type="button"
          onClick={onAddSources ?? onOpenSourceSelector}
          className="inline-flex h-7 items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2.5 text-[11px] font-medium text-primaryStrong hover:bg-primary/15 transition-colors"
        >
          <FolderPlus className="h-3.5 w-3.5" />
          Add sources
        </button>
      ) : null}

      {/* Sources pill */}
      <button
        type="button"
        onClick={onOpenSourceSelector}
        aria-label={sourceControlLabel}
        title="Open source scope and saved profiles"
        className="inline-flex h-7 items-center gap-1 rounded-full border border-border bg-surface px-2.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
      >
        <Layers className="h-3.5 w-3.5" />
        Sources: {sourceSummary}
        {specificSourceSummary ? (
          <span className="hidden sm:inline"> • Specific: {specificSourceSummary}</span>
        ) : null}
        <ChevronDown className="h-3 w-3" />
      </button>

      {/* Preset pill */}
      <button
        type="button"
        onClick={onOpenSettings}
        className="inline-flex h-7 items-center gap-1 rounded-full border border-border bg-surface px-2.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
        title={`Search preset: ${PRESET_LABELS[preset] ?? preset}`}
      >
        {PRESET_LABELS[preset] ?? preset}
        <ChevronDown className="h-3 w-3" />
      </button>

      {/* Web toggle pill */}
      <button
        type="button"
        onClick={onToggleWeb}
        disabled={!webFallbackAvailable}
        className={cn(
          "inline-flex h-7 items-center gap-1 rounded-full border px-2.5 text-[11px] font-medium transition-colors",
          webEnabled && webFallbackAvailable
            ? "border-primary/40 bg-primary/10 text-primary"
            : "border-border bg-surface text-text-muted hover:bg-surface2 hover:text-text",
          !webFallbackAvailable && "opacity-60 cursor-not-allowed hover:bg-surface hover:text-text-muted"
        )}
        aria-pressed={webEnabled && webFallbackAvailable}
        aria-label={
          webFallbackAvailable
            ? `Web fallback is currently ${webEnabled ? "enabled" : "disabled"}. Click to toggle.`
            : "Web fallback is not available on this server."
        }
        title={
          webFallbackAvailable
            ? "Falls back to web search when local source relevance is below threshold."
            : "Web fallback is not available on this server."
        }
      >
        <Globe className={cn("h-3.5 w-3.5", webEnabled && webFallbackAvailable ? "fill-current" : "")} />
        Web
      </button>

      <AnswerModelMenu
        generationProvider={generationProvider}
        generationModel={generationModel}
        onGenerationProviderChange={onGenerationProviderChange}
        onGenerationModelChange={onGenerationModelChange}
      />

      {sourceHealth && onRefreshSourceHealth ? (
        <button
          type="button"
          onClick={onRefreshSourceHealth}
          className="inline-flex h-7 items-center rounded-full border border-border bg-surface px-2.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
          aria-label="Refresh source health"
          title="Refresh source health"
        >
          {buildSourceHealthSummary(sourceHealth)}
        </button>
      ) : sourceHealth ? (
        <span className="inline-flex h-7 items-center rounded-full border border-border bg-surface px-2.5 text-[11px] font-medium text-text-muted">
          {buildSourceHealthSummary(sourceHealth)}
        </span>
      ) : null}

      {/* Settings gear */}
      <button
        type="button"
        onClick={onOpenSettings}
        className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-border text-text-muted hover:bg-surface2 hover:text-text transition-colors"
        aria-label="Open Knowledge QA settings"
        title="Open Knowledge QA settings"
      >
        <Settings className="h-3.5 w-3.5" />
      </button>

      {contextChangedSinceLastRun && (
        <Popover
          trigger="click"
          placement="bottomRight"
          title="Scope changed since last search"
          content={
            <div className="max-w-xs space-y-1.5">
              {scopeChangeDetails.length > 0 ? (
                <ul className="list-disc pl-4 text-xs text-text-muted space-y-1">
                  {scopeChangeDetails.map((detail, index) => (
                    <li key={index}>{detail}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-text-muted">
                  Search settings have changed since your last query.
                </p>
              )}
              <p className="text-xs text-text-muted pt-1 border-t border-border/60">
                Run a new search to apply the updated settings.
              </p>
            </div>
          }
        >
          <button
            type="button"
            className="inline-flex items-center rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary hover:bg-primary/20 transition-colors cursor-pointer"
          >
            Scope changed
          </button>
        </Popover>
      )}
    </div>
  )
}
