import React, { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import { BookOpen, ChevronDown, ChevronUp, CircleHelp, Clock3, FolderPlus, Globe, HelpCircle, MessageSquare, SlidersHorizontal } from "lucide-react"
import { cn } from "@/libs/utils"
import type { RagSource } from "@/services/rag/unified-rag"
import type { KnowledgeSourceHealthState } from "../types"
import type { KnowledgeReadyRecoveryState } from "./recoveryState"

type KnowledgeReadyStateProps = {
  suggestedPrompts: string[]
  onPromptClick: (prompt: string) => void
  onContinueRecent: () => void
  onSelectSources: () => void
  onAddSources?: () => void
  onEnableWebFallback?: () => void
  hasSources: boolean
  selectedSources?: RagSource[]
  sourceHealth?: KnowledgeSourceHealthState
  hasRecentSession: boolean
  webFallbackEnabled?: boolean
  recoveryState?: KnowledgeReadyRecoveryState
  className?: string
}

type SourceHealthNotice = {
  message: string
  tone: "info" | "warn"
  actionLabel: string
  action: "add" | "select"
}

function createFallbackRecoveryState(
  hasSources: boolean,
  webFallbackEnabled: boolean
): KnowledgeReadyRecoveryState {
  if (hasSources) {
    return {
      kind: "ready",
      hasIndexedSources: true,
      hasSelectedSources: true,
      webFallbackAvailable: true,
      webFallbackEnabled,
      canSearchPersonalLibrary: true,
      canSearchWebOnly: false,
      searchBlocked: false,
    }
  }

  return {
    kind: webFallbackEnabled ? "web_only" : "no_selected_sources",
    hasIndexedSources: true,
    hasSelectedSources: false,
    webFallbackAvailable: true,
    webFallbackEnabled,
    canSearchPersonalLibrary: false,
    canSearchWebOnly: webFallbackEnabled,
    searchBlocked: !webFallbackEnabled,
  }
}

function buildSourceHealthNotice(
  hasSources: boolean,
  selectedSources: RagSource[],
  sourceHealth: KnowledgeSourceHealthState | undefined
): SourceHealthNotice | null {
  if (!hasSources) return null
  if (sourceHealth?.error) {
    return {
      message: sourceHealth.error,
      tone: "info",
      actionLabel: "Select sources",
      action: "select",
    }
  }
  if (!sourceHealth || selectedSources.length === 0) return null

  const selectedHealth = selectedSources
    .map((source) => sourceHealth.bySource[source])
    .filter((source) => source != null)
  if (selectedHealth.length === 0) return null

  const allSelectedUnavailable = selectedHealth.every(
    (source) =>
      !source.available ||
      source.indexStatus === "unavailable" ||
      source.indexStatus === "error"
  )
  if (allSelectedUnavailable) {
    return {
      message: "Selected sources are unavailable. Open source settings or choose a different scope.",
      tone: "warn",
      actionLabel: "Open source settings",
      action: "select",
    }
  }

  const allSelectedEmpty = selectedHealth.every(
    (source) => source.available && source.indexStatus === "empty"
  )
  if (allSelectedEmpty) {
    return {
      message: "No searchable items yet. Open Quick Ingest or the source owner page to add content.",
      tone: "warn",
      actionLabel: "Open Quick Ingest",
      action: "add",
    }
  }

  return null
}

export function KnowledgeReadyState({
  suggestedPrompts,
  onPromptClick,
  onContinueRecent,
  onSelectSources,
  onAddSources,
  onEnableWebFallback,
  hasSources,
  selectedSources = [],
  sourceHealth,
  hasRecentSession,
  webFallbackEnabled = false,
  recoveryState,
  className,
}: KnowledgeReadyStateProps) {
  const isReturningUser = hasRecentSession
  const [guideExpanded, setGuideExpanded] = useState(!isReturningUser)
  const handleAddSources = onAddSources ?? onSelectSources
  const effectiveRecoveryState =
    recoveryState ?? createFallbackRecoveryState(hasSources, webFallbackEnabled)
  const hasSelectedSources = effectiveRecoveryState.hasSelectedSources
  const shouldUseAddSourceAction =
    !recoveryState && (!hasSources || !effectiveRecoveryState.hasIndexedSources)
  const sourceHealthNotice = buildSourceHealthNotice(
    hasSources,
    selectedSources,
    sourceHealth
  )
  const showRecoveryNotice = effectiveRecoveryState.kind !== "ready"
  const canOfferWebFallback =
    effectiveRecoveryState.webFallbackAvailable &&
    !effectiveRecoveryState.webFallbackEnabled &&
    typeof onEnableWebFallback === "function"
  const recoveryTone = effectiveRecoveryState.searchBlocked ? "warn" : "info"
  const recoveryTitle = (() => {
    switch (effectiveRecoveryState.kind) {
      case "backend_unavailable":
        return "Library search is offline"
      case "no_indexed_sources":
      case "no_indexed_sources_web_only":
        return "No indexed library sources yet"
      case "no_selected_sources":
        return "No source categories selected"
      case "web_only":
        return "Web-only search"
      default:
        return null
    }
  })()
  const recoveryBody = (() => {
    switch (effectiveRecoveryState.kind) {
      case "backend_unavailable":
        return "The Knowledge QA backend is not reachable, so cited library answers cannot run yet."
      case "no_indexed_sources":
        return "Your server is online, but Knowledge QA has no indexed documents, media, or notes to search."
      case "no_indexed_sources_web_only":
        return "Your personal library has no indexed sources yet. Because web fallback is enabled, searches will use web results only until you add or index sources."
      case "no_selected_sources":
        return effectiveRecoveryState.webFallbackAvailable
          ? "Your library has indexed sources, but none are selected for this search."
          : "Your library has indexed sources, but none are selected and web fallback is not available on this server."
      case "web_only":
        return "No source categories are selected. Because web fallback is enabled, this search will use web results only and will not cite your personal library until sources are selected."
      default:
        return null
    }
  })()

  // Collapse guide when history finishes loading and reveals a returning user
  useEffect(() => {
    if (isReturningUser) {
      setGuideExpanded(false)
    }
  }, [isReturningUser])

  return (
    <div className={cn("space-y-5 text-center", className)}>
      <div className="mx-auto max-w-2xl">
        <BookOpen className="mx-auto mb-3 h-12 w-12 text-primary" />
        <h1 className="text-3xl font-bold">Ask Your Library</h1>
        {!guideExpanded && (
          <button
            type="button"
            onClick={() => setGuideExpanded(true)}
            className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text transition-colors"
            title="How it works"
          >
            <CircleHelp className="h-3.5 w-3.5" />
            <span>How it works</span>
          </button>
        )}
        <p className="mt-1 text-base font-medium">
          Search selected personal-library sources and get cited answers
        </p>
        <p className="mt-2 text-sm text-text-muted">
          Knowledge QA searches documents, media, notes, and other selected sources, then grounds answer claims in citations you can inspect.
        </p>
        <p className="mt-2 text-sm text-text-muted">
          This page answers questions over searchable sources. Add or manage sources
          in their owner pages, then use /knowledge for QA.
        </p>
      </div>

      {/* How it works - adapts to user state */}
      <div className="mx-auto max-w-2xl rounded-lg border border-border/80 bg-surface2/60 px-4 py-3 text-left">
        <button
          type="button"
          onClick={() => setGuideExpanded((prev) => !prev)}
          className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wide text-text-muted"
          aria-expanded={guideExpanded}
        >
          <span>How it works</span>
          {isReturningUser && (
            guideExpanded
              ? <ChevronUp className="h-3.5 w-3.5" />
              : <ChevronDown className="h-3.5 w-3.5" />
          )}
        </button>
        {guideExpanded && (
          isReturningUser ? (
            <p className="mt-2 text-sm text-text-muted">
              Select sources, ask a question, review cited answers.
            </p>
          ) : !effectiveRecoveryState.hasIndexedSources ? (
            <ol className="mt-2 grid gap-1 text-sm text-text-muted sm:grid-cols-3 sm:gap-3">
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-white">1</span>
                <span>
                  <FolderPlus className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-primary" />
                  <button
                    type="button"
                    onClick={handleAddSources}
                    className="font-medium text-primary hover:underline"
                  >
                    Add sources
                  </button>
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-text/20 text-[10px] font-bold text-text">2</span>
                <span>
                  <MessageSquare className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-text-muted" />
                  Ask a question
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-text/20 text-[10px] font-bold text-text">3</span>
                <span>
                  <HelpCircle className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-text-muted" />
                  Review cited answer
                </span>
              </li>
            </ol>
          ) : (
            <ol className="mt-2 grid gap-1 text-sm text-text-muted sm:grid-cols-3 sm:gap-3">
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-white">1</span>
                <span>
                  <SlidersHorizontal className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-primary" />
                  {hasSelectedSources ? "Select sources" : (
                    <button
                      type="button"
                      onClick={onSelectSources}
                      className="font-medium text-primary hover:underline"
                    >
                      Select source categories
                    </button>
                  )}
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-text/20 text-[10px] font-bold text-text">2</span>
                <span>
                  <MessageSquare className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-text-muted" />
                  Ask a question
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-text/20 text-[10px] font-bold text-text">3</span>
                <span>
                  <HelpCircle className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-text-muted" />
                  Review cited answer
                </span>
              </li>
            </ol>
          )
        )}
      </div>

      <div className="mx-auto flex max-w-2xl flex-wrap items-center justify-center gap-2">
        {suggestedPrompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onPromptClick(prompt)}
            className="rounded-md border border-border/80 bg-surface2/70 px-3 py-1.5 text-[11px] text-text transition-colors hover:border-primary/40 hover:bg-surface2"
          >
            {prompt}
          </button>
        ))}
      </div>

      {showRecoveryNotice && recoveryTitle && recoveryBody ? (
        <div
          className={cn(
            "mx-auto max-w-2xl rounded-lg px-4 py-3 text-left text-sm",
            recoveryTone === "info"
              ? "border border-info/30 bg-info/10 text-info"
              : "border border-warn/30 bg-warn/10 text-warn"
          )}
        >
          <p className="font-medium">{recoveryTitle}</p>
          <p className="mt-1">{recoveryBody}</p>
          {effectiveRecoveryState.webFallbackEnabled ? (
            <>
              <p className="mt-1">
                Web fallback uses your configured server default provider.
              </p>
              <p className="mt-1">
                Queries stay on your tldw server unless web fallback is enabled.
              </p>
            </>
          ) : null}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {effectiveRecoveryState.kind === "no_indexed_sources" ||
            effectiveRecoveryState.kind === "no_indexed_sources_web_only" ? (
              <>
                <button
                  type="button"
                  onClick={handleAddSources}
                  className={cn(
                    "inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                    recoveryTone === "info"
                      ? "border-info/40 hover:bg-info/20"
                      : "border-warn/40 hover:bg-warn/20"
                  )}
                >
                  Add or index sources
                </button>
                <Link
                  to="/notes"
                  className={cn(
                    "inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                    recoveryTone === "info"
                      ? "border-info/40 hover:bg-info/20"
                      : "border-warn/40 hover:bg-warn/20"
                  )}
                >
                  Create a note
                </Link>
              </>
            ) : (
              <button
                type="button"
                onClick={onSelectSources}
                className={cn(
                  "inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                  recoveryTone === "info"
                    ? "border-info/40 hover:bg-info/20"
                    : "border-warn/40 hover:bg-warn/20"
                )}
              >
                Select source categories
              </button>
            )}
            {canOfferWebFallback ? (
              <button
                type="button"
                onClick={onEnableWebFallback}
                className={cn(
                  "inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                  recoveryTone === "info"
                    ? "border-info/40 hover:bg-info/20"
                    : "border-warn/40 hover:bg-warn/20"
                )}
              >
                <Globe className="h-3.5 w-3.5" />
                Enable web fallback
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      {sourceHealthNotice ? (
        <div
          className={cn(
            "mx-auto max-w-2xl rounded-lg px-4 py-3 text-left text-sm",
            sourceHealthNotice.tone === "info"
              ? "border border-info/30 bg-info/10 text-info"
              : "border border-warn/30 bg-warn/10 text-warn"
          )}
        >
          <p>{sourceHealthNotice.message}</p>
          <button
            type="button"
            onClick={
              sourceHealthNotice.action === "select"
                ? onSelectSources
                : handleAddSources
            }
            className={cn(
              "mt-2 inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
              sourceHealthNotice.tone === "info"
                ? "border-info/40 hover:bg-info/20"
                : "border-warn/40 hover:bg-warn/20"
            )}
          >
            {sourceHealthNotice.actionLabel}
          </button>
        </div>
      ) : null}

      <div className="mx-auto flex max-w-2xl flex-wrap items-center justify-center gap-2">
        <p className="basis-full text-xs text-text-muted">
          {hasRecentSession
            ? "Recent QA session available."
            : "No previous QA sessions yet."}
        </p>
        <button
          type="button"
          onClick={onContinueRecent}
          disabled={!hasRecentSession}
          className={cn(
            "inline-flex h-8 items-center gap-1 rounded-md border px-3 text-sm transition-colors",
            hasRecentSession
              ? "border-border text-text hover:bg-surface2"
              : "border-border text-text-subtle cursor-not-allowed opacity-70"
          )}
        >
          <Clock3 className="h-4 w-4" />
          Continue recent session
        </button>
        <button
          type="button"
          onClick={shouldUseAddSourceAction ? handleAddSources : onSelectSources}
          className={cn(
            "inline-flex h-8 items-center gap-1 rounded-md border px-3 text-sm transition-colors",
            hasSelectedSources
              ? "border-border text-text hover:bg-surface2"
              : "border-warn/40 bg-warn/10 text-warn hover:bg-warn/20"
          )}
        >
          <SlidersHorizontal className="h-4 w-4" />
          {hasSelectedSources
            ? "Select sources"
            : shouldUseAddSourceAction
              ? "Add sources"
              : "No sources selected"}
        </button>
      </div>

      <p className="text-[11px] text-text-subtle">
        Need a full workspace?{" "}
        <Link to="/research-workspace" className="text-primary/70 hover:text-primary transition-colors">
          Try Research Workspace &rarr;
        </Link>
      </p>
    </div>
  )
}
