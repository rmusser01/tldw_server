/**
 * SearchBar - Prominent search input for Knowledge QA
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Tooltip } from "antd"
import { CloudOff, Globe, Search, Sparkles, Square, X } from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import { trackKnowledgeQaSearchMetric } from "@/utils/knowledge-qa-search-metrics"
import {
  buildLocalQuerySuggestions,
  shouldShowSuggestionPrototype,
  type QuerySuggestion,
} from "./querySuggestions"

const EXAMPLE_QUERIES = [
  "What are the key findings from the research?",
  "Summarize the main arguments in this document",
  "What are the pros and cons discussed?",
  "Explain the methodology used in this study",
  "Compare the conclusions across my PDFs",
  "When was topic X first mentioned in my sources?",
  "What are the citations for claim Y?",
]
const MAX_QUERY_LENGTH = 20000
const SUGGESTION_SOURCE_LABELS: Record<QuerySuggestion["source"], string> = {
  history: "History",
  source_title: "Source",
  example: "Example",
}

type SearchBarProps = {
  className?: string
  autoFocus?: boolean
  showWebToggle?: boolean
  widthMode?: "compact" | "wide"
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tagName = target.tagName.toUpperCase()
  return (
    tagName === "INPUT" ||
    tagName === "TEXTAREA" ||
    tagName === "SELECT" ||
    target.isContentEditable
  )
}

function isInteractiveControlTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return Boolean(
    target.closest(
      [
        "button",
        "a[href]",
        "input",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
        "[role='menuitem']",
        "[role='option']",
        "[role='switch']",
        "[role='tab']",
      ].join(",")
    )
  )
}

function isKnowledgeSearchInputTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement && target.id === "knowledge-search-input"
}

export function SearchBar({
  className,
  autoFocus = true,
  showWebToggle = true,
  widthMode = "compact",
}: SearchBarProps) {
  const {
    query,
    setQuery,
    search,
    cancelSearch,
    isSearching,
    clearResults,
    results,
    answer,
    queryWarning,
    searchHistory,
    isLocalOnlyThread,
    settings,
    updateSetting,
  } = useKnowledgeQA()
  const inputRef = useRef<HTMLInputElement>(null)
  const blurTimeoutRef = useRef<number | null>(null)
  const [placeholderIndex, setPlaceholderIndex] = useState(0)
  const [isFocused, setIsFocused] = useState(false)
  const [cycleCount, setCycleCount] = useState(0)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(-1)
  const hasResults = results.length > 0 || Boolean(answer)
  const hasSources = settings.sources.length > 0
  const noSourcesBlocked = !hasSources && !settings.enable_web_fallback
  const showHintEmphasis = !query && !isSearching && cycleCount === 0
  const showCharacterCount = query.length >= Math.floor(MAX_QUERY_LENGTH * 0.8)
  const historyQueries = useMemo(
    () => searchHistory.map((item) => item.query).filter((item) => item.trim().length > 0),
    [searchHistory]
  )
  const sourceTitles = useMemo(
    () =>
      results
        .map((result) =>
          typeof result.metadata?.title === "string" ? result.metadata.title : ""
        )
        .filter((title) => title.trim().length > 0),
    [results]
  )
  const suggestions = useMemo(
    () =>
      buildLocalQuerySuggestions({
        query,
        historyQueries,
        sourceTitles,
        exampleQueries: EXAMPLE_QUERIES,
        limit: 5,
      }),
    [query, historyQueries, sourceTitles]
  )
  const shouldShowSuggestions =
    showSuggestions &&
    isFocused &&
    !isSearching &&
    shouldShowSuggestionPrototype(query) &&
    suggestions.length > 0

  // Rotate placeholder examples continuously while idle; pause on focus or when typing.
  useEffect(() => {
    if (isFocused || query) return

    const interval = setInterval(() => {
      setPlaceholderIndex((prev) => {
        const next = (prev + 1) % EXAMPLE_QUERIES.length
        // Increment cycle count when we loop back to start
        if (next === 0) {
          setCycleCount((c) => c + 1)
        }
        return next
      })
    }, 8000) // 8 seconds instead of 4 for better readability

    return () => clearInterval(interval)
  }, [isFocused, query])

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const targetIsEditable = isEditableTarget(e.target)
      const targetIsInteractiveControl = isInteractiveControlTarget(e.target)
      const targetIsKnowledgeSearchInput = isKnowledgeSearchInputTarget(e.target)

      // Focus search bar on "/" key
      if (e.key === "/" && !e.metaKey && !e.ctrlKey) {
        if (!targetIsEditable) {
          e.preventDefault()
          inputRef.current?.focus()
        }
      }
      // Cmd+K for new search
      if (
        (e.metaKey || e.ctrlKey) &&
        e.key.toLowerCase() === "k" &&
        (!targetIsEditable || targetIsKnowledgeSearchInput) &&
        (!targetIsInteractiveControl || targetIsKnowledgeSearchInput)
      ) {
        e.preventDefault()
        clearResults()
        setQuery("")
        inputRef.current?.focus()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [clearResults, setQuery])

  // Auto focus on mount
  useEffect(() => {
    if (autoFocus) {
      inputRef.current?.focus()
    }
  }, [autoFocus])

  useEffect(
    () => () => {
      if (blurTimeoutRef.current) {
        window.clearTimeout(blurTimeoutRef.current)
      }
    },
    []
  )

  useEffect(() => {
    if (!shouldShowSuggestions) {
      setActiveSuggestionIndex(-1)
      return
    }
    setActiveSuggestionIndex((previous) => {
      if (previous < 0) {
        return previous
      }
      if (suggestions.length === 0) {
        return -1
      }
      return Math.min(previous, suggestions.length - 1)
    })
  }, [shouldShowSuggestions, suggestions.length])

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      if (query.trim() && !isSearching && !noSourcesBlocked) {
        setShowSuggestions(false)
        search()
      }
    },
    [query, isSearching, noSourcesBlocked, search]
  )

  const applySuggestion = useCallback(
    (suggestion: QuerySuggestion) => {
      setQuery(suggestion.text)
      void trackKnowledgeQaSearchMetric({
        type: "suggestion_accept",
        source: suggestion.source,
      })
      setShowSuggestions(false)
      setActiveSuggestionIndex(-1)
      requestAnimationFrame(() => inputRef.current?.focus())
    },
    [setQuery]
  )

  const handleClear = useCallback(() => {
    setQuery("")
    setShowSuggestions(true)
    inputRef.current?.focus()
  }, [setQuery])

  const handleNewSearch = useCallback(() => {
    clearResults()
    setQuery("")
    inputRef.current?.focus()
  }, [clearResults, setQuery])

  return (
    <form
      onSubmit={handleSubmit}
      className={cn("mx-auto w-full", widthMode === "compact" && "max-w-3xl", className)}
    >
      <p id="knowledge-qa-search-description" className="sr-only">
        Ask questions about your documents and get AI-powered answers with citations from your knowledge base.
      </p>
      <div className="relative group">
        {/* Search icon */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none">
          {isSearching ? (
            <div className="animate-spin">
              <Sparkles className="w-5 h-5 text-primary" />
            </div>
          ) : (
            <Search className="w-5 h-5 text-text-muted group-focus-within:text-primary transition-colors" />
          )}
        </div>

        {/* Input */}
        <input
          id="knowledge-search-input"
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value.slice(0, MAX_QUERY_LENGTH))}
          onFocus={() => {
            if (blurTimeoutRef.current) {
              window.clearTimeout(blurTimeoutRef.current)
              blurTimeoutRef.current = null
            }
            setIsFocused(true)
            setShowSuggestions(true)
          }}
          onBlur={() => {
            setIsFocused(false)
            blurTimeoutRef.current = window.setTimeout(() => {
              setShowSuggestions(false)
              setActiveSuggestionIndex(-1)
            }, 120)
          }}
          onKeyDown={(e) => {
            if (!shouldShowSuggestions) return

            if (e.key === "ArrowDown") {
              e.preventDefault()
              setActiveSuggestionIndex((prev) =>
                prev >= suggestions.length - 1 ? 0 : prev + 1
              )
              return
            }
            if (e.key === "ArrowUp") {
              e.preventDefault()
              setActiveSuggestionIndex((prev) =>
                prev <= 0 ? suggestions.length - 1 : prev - 1
              )
              return
            }
            if (e.key === "Enter" && activeSuggestionIndex >= 0) {
              const activeSuggestion = suggestions[activeSuggestionIndex]
              if (!activeSuggestion) {
                e.preventDefault()
                setActiveSuggestionIndex(
                  suggestions.length > 0 ? suggestions.length - 1 : -1
                )
                return
              }
              e.preventDefault()
              applySuggestion(activeSuggestion)
              return
            }
            if (e.key === "Escape") {
              e.preventDefault()
              setShowSuggestions(false)
              setActiveSuggestionIndex(-1)
            }
          }}
          placeholder={EXAMPLE_QUERIES[placeholderIndex]}
          disabled={isSearching}
          maxLength={MAX_QUERY_LENGTH}
          className={cn(
            "w-full pl-12 pr-20 py-4 text-lg",
            "bg-surface border border-border rounded-xl",
            "focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent",
            "placeholder:text-text-subtle",
            "transition-all duration-200",
            "shadow-sm hover:shadow-md focus:shadow-md",
            isSearching && "opacity-75 cursor-not-allowed"
          )}
          aria-label="Search your knowledge base"
          aria-describedby="knowledge-qa-search-description"
          aria-autocomplete="list"
          aria-expanded={shouldShowSuggestions}
          aria-controls={shouldShowSuggestions ? "knowledge-search-suggestions" : undefined}
        />

        {shouldShowSuggestions && (
          <div
            id="knowledge-search-suggestions"
            role="listbox"
            className="absolute left-0 right-0 top-full z-20 mt-2 overflow-hidden rounded-xl border border-border bg-surface shadow-lg"
          >
            {suggestions.map((suggestion, index) => (
              <button
                key={suggestion.id}
                type="button"
                role="option"
                aria-selected={activeSuggestionIndex === index}
                className={cn(
                  "flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm transition-colors",
                  activeSuggestionIndex === index
                    ? "bg-primary/10 text-primary"
                    : "hover:bg-hover"
                )}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => applySuggestion(suggestion)}
              >
                <span className="truncate">{suggestion.text}</span>
                <span className="text-[10px] uppercase tracking-wide text-text-subtle">
                  {SUGGESTION_SOURCE_LABELS[suggestion.source]}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Clear button */}
        {query && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-14 top-1/2 -translate-y-1/2 p-1 text-text-muted hover:text-text transition-colors"
            aria-label="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        )}

        {/* Submit button */}
        <Tooltip
          title={
            noSourcesBlocked
              ? "Select source categories or enable web fallback to search"
              : null
          }
        >
          <span className="absolute right-2 top-1/2 inline-flex -translate-y-1/2">
            <button
              type="submit"
              disabled={!query.trim() || isSearching || noSourcesBlocked}
              className={cn(
                "px-4 py-2 rounded-lg",
                "bg-primary text-white",
                "font-medium text-sm",
                "transition-all duration-200",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                "hover:bg-primaryStrong"
              )}
            >
              {isSearching ? "Searching..." : "Ask"}
            </button>
          </span>
        </Tooltip>
      </div>

      {isSearching && (
        <div
          className="mt-2 h-1 w-full overflow-hidden rounded-full bg-bg-subtle"
          data-testid="knowledge-search-loading-indicator"
        >
          <div className="h-full w-1/3 rounded-full bg-primary animate-pulse" />
        </div>
      )}

      {queryWarning ? (
        <p
          role="status"
          aria-live="polite"
          className="mt-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn"
        >
          {queryWarning}
        </p>
      ) : null}

      {/* Keyboard hint + quick controls */}
      <div
        className={cn(
          "mt-2 flex items-center justify-between gap-3 text-xs text-text-muted",
          showHintEmphasis && "text-sm text-text"
        )}
      >
        <span className="truncate">
          Press <kbd className="px-1.5 py-0.5 rounded font-mono text-text bg-gray-100 dark:bg-gray-800 border border-border/60">/</kbd> to focus,{" "}
          <kbd className="px-1.5 py-0.5 rounded font-mono text-text bg-gray-100 dark:bg-gray-800 border border-border/60">Cmd+K</kbd> for new search
        </span>
        <div className="flex items-center gap-2">
          {isLocalOnlyThread && (
            <span
              className="inline-flex items-center gap-1 rounded-md border border-warn/40 bg-warn/10 px-2 py-1 text-[11px] text-warn"
              title="Working offline - conversation is stored locally and not synced to server."
            >
              <CloudOff className="h-3 w-3" />
              Not synced
            </span>
          )}
          {hasResults && (
            <button
              type="button"
              onClick={handleNewSearch}
              className="px-2 py-1 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors whitespace-nowrap"
              title="Clear current results and start a new search"
            >
              New search
            </button>
          )}
          {isSearching && (
            <Tooltip title="Stop search. Partial results will be kept if available.">
              <button
                type="button"
                onClick={cancelSearch}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-warn/40 bg-warn/10 text-warn hover:bg-warn/20 transition-colors whitespace-nowrap"
                aria-label="Cancel search"
              >
                <Square className="w-3 h-3" />
                Stop
              </button>
            </Tooltip>
          )}
          {showWebToggle ? (
            <button
              type="button"
              onClick={() =>
                updateSetting("enable_web_fallback", !settings.enable_web_fallback)
              }
              className={cn(
                "inline-flex items-center gap-1 px-2 py-1 rounded-full border transition-colors whitespace-nowrap",
                settings.enable_web_fallback
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border bg-surface text-text-subtle hover:bg-hover hover:text-text"
              )}
              aria-pressed={settings.enable_web_fallback}
              title="Falls back to web search when local source relevance is below threshold (configurable in settings)."
              aria-label={`Web fallback is currently ${settings.enable_web_fallback ? "enabled" : "disabled"}. Click to toggle.`}
            >
              <Globe className="w-3.5 h-3.5" />
              Web
            </button>
          ) : null}
        </div>
      </div>

      {showCharacterCount && (
        <div className="mt-1 text-right text-xs text-text-muted">
          {query.length}/{MAX_QUERY_LENGTH}
        </div>
      )}
    </form>
  )
}
