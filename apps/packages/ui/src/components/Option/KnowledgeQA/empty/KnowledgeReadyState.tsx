import React, { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import { BookOpen, ChevronDown, ChevronUp, CircleHelp, Clock3, FolderPlus, HelpCircle, MessageSquare, SlidersHorizontal } from "lucide-react"
import { cn } from "@/libs/utils"

type KnowledgeReadyStateProps = {
  suggestedPrompts: string[]
  onPromptClick: (prompt: string) => void
  onContinueRecent: () => void
  onSelectSources: () => void
  hasSources: boolean
  hasRecentSession: boolean
  webFallbackEnabled?: boolean
  className?: string
}

export function KnowledgeReadyState({
  suggestedPrompts,
  onPromptClick,
  onContinueRecent,
  onSelectSources,
  hasSources,
  hasRecentSession,
  webFallbackEnabled = false,
  className,
}: KnowledgeReadyStateProps) {
  const isReturningUser = hasRecentSession
  const [guideExpanded, setGuideExpanded] = useState(!isReturningUser)

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
        <p className="mt-1 text-base font-medium">Search your documents and get cited answers</p>
        <p className="mt-2 text-sm text-text-muted">
          Get grounded answers with citations from your selected sources.
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
          ) : !hasSources ? (
            <ol className="mt-2 grid gap-1 text-sm text-text-muted sm:grid-cols-3 sm:gap-3">
              <li className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-white">1</span>
                <span>
                  <FolderPlus className="mb-0.5 mr-1 inline h-3.5 w-3.5 text-primary" />
                  <button
                    type="button"
                    onClick={onSelectSources}
                    className="font-medium text-primary hover:underline"
                  >
                    Add documents first
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
                  Select sources
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

      {!hasSources ? (
        <div
          className={cn(
            "mx-auto max-w-2xl rounded-lg px-4 py-3 text-left text-sm",
            webFallbackEnabled
              ? "border border-info/30 bg-info/10 text-info"
              : "border border-warn/30 bg-warn/10 text-warn"
          )}
        >
          <p>
            {webFallbackEnabled
              ? "No document sources are selected. Your search will use web results only."
              : "No sources are selected. Select source categories to search, or enable web fallback."}
          </p>
          <button
            type="button"
            onClick={onSelectSources}
            className={cn(
              "mt-2 inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
              webFallbackEnabled
                ? "border-info/40 hover:bg-info/20"
                : "border-warn/40 hover:bg-warn/20"
            )}
          >
            Open source settings
          </button>
        </div>
      ) : null}

      <div className="mx-auto flex max-w-2xl flex-wrap items-center justify-center gap-2">
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
          onClick={onSelectSources}
          className={cn(
            "inline-flex h-8 items-center gap-1 rounded-md border px-3 text-sm transition-colors",
            hasSources
              ? "border-border text-text hover:bg-surface2"
              : "border-warn/40 bg-warn/10 text-warn hover:bg-warn/20"
          )}
        >
          <SlidersHorizontal className="h-4 w-4" />
          {hasSources ? "Select sources" : "No sources selected"}
        </button>
      </div>

      <p className="text-[11px] text-text-subtle">
        Need a full workspace?{" "}
        <Link to="/workspace-playground" className="text-primary/70 hover:text-primary transition-colors">
          Try Research Studio &rarr;
        </Link>
      </p>
    </div>
  )
}
