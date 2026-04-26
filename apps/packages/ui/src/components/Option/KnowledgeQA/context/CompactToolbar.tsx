import React from "react"
import { Popover } from "antd"
import { Layers, Globe, ChevronDown, Settings } from "lucide-react"
import { cn } from "@/libs/utils"
import type { RagPresetName, RagSource } from "@/services/rag/unified-rag"
import { ALL_RAG_SOURCES, getRagSourceLabel } from "@/services/rag/sourceMetadata"
import { AnswerModelMenu } from "./AnswerModelMenu"

type CompactToolbarProps = {
  sources: RagSource[]
  preset: RagPresetName
  webEnabled: boolean
  onToggleWeb: () => void
  onOpenSourceSelector: () => void
  onOpenSettings: () => void
  generationProvider: string | null
  generationModel: string | null
  onGenerationProviderChange: (provider: string | null) => void
  onGenerationModelChange: (model: string | null) => void
  contextChangedSinceLastRun: boolean
  scopeChangeDetails?: string[]
  className?: string
}

const ALL_SOURCES_THRESHOLD = ALL_RAG_SOURCES.length

function summarizeSources(sources: RagSource[]): string {
  if (!Array.isArray(sources) || sources.length === 0) return "None"
  if (sources.length === 1) return getRagSourceLabel(sources[0])
  if (sources.length >= ALL_SOURCES_THRESHOLD) return "All sources"
  return `${sources.length} selected`
}

const PRESET_LABELS: Record<string, string> = {
  fast: "Fast",
  balanced: "Balanced",
  thorough: "Deep",
  custom: "Custom",
}

export function CompactToolbar({
  sources,
  preset,
  webEnabled,
  onToggleWeb,
  onOpenSourceSelector,
  onOpenSettings,
  generationProvider,
  generationModel,
  onGenerationProviderChange,
  onGenerationModelChange,
  contextChangedSinceLastRun,
  scopeChangeDetails = [],
  className,
}: CompactToolbarProps) {
  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      {/* Sources pill */}
      <button
        type="button"
        onClick={onOpenSourceSelector}
        className="inline-flex h-7 items-center gap-1 rounded-full border border-border bg-surface px-2.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
      >
        <Layers className="h-3.5 w-3.5" />
        Sources: {summarizeSources(sources)}
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
        className={cn(
          "inline-flex h-7 items-center gap-1 rounded-full border px-2.5 text-[11px] font-medium transition-colors",
          webEnabled
            ? "border-primary/40 bg-primary/10 text-primary"
            : "border-border bg-surface text-text-muted hover:bg-surface2 hover:text-text"
        )}
        aria-pressed={webEnabled}
        aria-label={`Web fallback is currently ${webEnabled ? "enabled" : "disabled"}. Click to toggle.`}
        title="Falls back to web search when local source relevance is below threshold."
      >
        <Globe className={cn("h-3.5 w-3.5", webEnabled ? "fill-current" : "")} />
        Web
      </button>

      <AnswerModelMenu
        generationProvider={generationProvider}
        generationModel={generationModel}
        onGenerationProviderChange={onGenerationProviderChange}
        onGenerationModelChange={onGenerationModelChange}
      />

      {/* Settings gear */}
      <button
        type="button"
        onClick={onOpenSettings}
        className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-border text-text-muted hover:bg-surface2 hover:text-text transition-colors"
        aria-label="Open settings"
        title="Open search settings"
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
