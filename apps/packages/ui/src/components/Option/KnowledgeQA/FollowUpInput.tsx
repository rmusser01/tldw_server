/**
 * FollowUpInput - Input for asking follow-up questions
 */

import React, { useCallback, useState } from "react"
import { SendHorizontal, Plus, Loader2 } from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import { useMobile } from "@/hooks/useMediaQuery"

type FollowUpInputProps = {
  className?: string
  mode?: "default" | "recovery"
}

const MAX_FOLLOW_UP_LENGTH = 20000

export function FollowUpInput({
  className,
  mode = "default",
}: FollowUpInputProps) {
  const {
    askFollowUp,
    isSearching,
    startNewTopic,
    results,
    answer,
    setEvidenceRailOpen,
  } = useKnowledgeQA()
  const isMobile = useMobile()
  const [input, setInput] = useState("")
  const [pendingAction, setPendingAction] = useState<"followup" | "new-topic" | null>(null)
  const shouldShow = isSearching || results.length > 0 || Boolean(answer)
  const isQueuedState = isSearching && results.length === 0 && !answer
  const useStickyMobileLayout = isMobile
  const controlsDisabled = isSearching || pendingAction !== null
  const showCharacterCount = input.length >= Math.floor(MAX_FOLLOW_UP_LENGTH * 0.8)
  const hitCharacterLimit = input.length >= MAX_FOLLOW_UP_LENGTH
  const promptTitle =
    mode === "recovery" && !isQueuedState ? "Try a sharper follow-up" : null
  const helperText = isQueuedState
    ? "Follow-up input unlocks when the current search completes."
    : mode === "recovery"
      ? "Ask for the missing detail, timeframe, or source you still need."
      : 'Follow-up questions maintain context. Click "New Topic" to start fresh.'
  const inputPlaceholder = isQueuedState
    ? "Current search in progress..."
    : mode === "recovery"
      ? "Ask a more specific follow-up..."
      : "Ask a follow-up question..."

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      const trimmed = input.trim()
      if (!trimmed || controlsDisabled) return

      setPendingAction("followup")
      try {
        await askFollowUp(trimmed)
        setInput("")
      } finally {
        setPendingAction(null)
      }
    },
    [askFollowUp, controlsDisabled, input],
  )

  const handleNewTopic = useCallback(async () => {
    if (controlsDisabled) return

    setPendingAction("new-topic")
    try {
      await startNewTopic()
      setInput("")
    } finally {
      setPendingAction(null)
    }
  }, [controlsDisabled, startNewTopic])

  if (!shouldShow) {
    return null
  }

  return (
    <>
      {useStickyMobileLayout && <div aria-hidden="true" className="h-44" />}
      <div
        data-testid={useStickyMobileLayout ? "knowledge-followup-sticky" : undefined}
        className={cn(
          useStickyMobileLayout
            ? "fixed inset-x-0 bottom-0 z-30 border-t border-border bg-surface/95 px-3 pt-2 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur supports-[backdrop-filter]:bg-surface/80"
            : "border-t border-border pt-4",
          className,
        )}
      >
        {promptTitle ? (
          <div className={cn("mb-3", useStickyMobileLayout && "mx-auto max-w-4xl")}>
            <p className="text-sm font-semibold text-text">{promptTitle}</p>
            <p className="mt-1 text-xs text-text-muted">{helperText}</p>
          </div>
        ) : null}
        <form
          onSubmit={handleSubmit}
          className={cn(
            "flex flex-wrap items-center gap-2",
            useStickyMobileLayout && "mx-auto max-w-4xl",
          )}
        >
          {/* Input */}
          <div
            className={cn(
              "relative min-w-0 flex-1",
              useStickyMobileLayout && "basis-full",
            )}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value.slice(0, MAX_FOLLOW_UP_LENGTH))}
              placeholder={inputPlaceholder}
              aria-label="Ask a follow-up question"
              disabled={controlsDisabled}
              maxLength={MAX_FOLLOW_UP_LENGTH}
              className={cn(
                "w-full pl-4 pr-12 py-3",
                "bg-surface border border-border rounded-lg",
                "focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent",
                "placeholder:text-text-subtle",
                "transition-all duration-200",
                controlsDisabled && "opacity-75 cursor-not-allowed",
              )}
            />

            {/* Submit button */}
            <button
              type="submit"
              aria-label="Submit follow-up question"
              disabled={!input.trim() || controlsDisabled}
              className={cn(
                "absolute right-2 top-1/2 -translate-y-1/2",
                "p-2 rounded-md",
                "bg-primary text-white",
                "transition-all duration-200",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                "hover:bg-primaryStrong",
              )}
            >
              {controlsDisabled ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <SendHorizontal className="w-4 h-4" />
              )}
            </button>
          </div>

          {useStickyMobileLayout && setEvidenceRailOpen ? (
            <button
              type="button"
              onClick={() => setEvidenceRailOpen(true)}
              aria-label="Open evidence panel"
              className="h-10 rounded-lg border border-border px-3 text-sm"
            >
              Evidence
            </button>
          ) : null}

          {/* New topic button */}
          <button
            type="button"
            onClick={handleNewTopic}
            aria-label="Start new topic"
            disabled={controlsDisabled}
            className="flex items-center gap-1.5 h-10 px-3 rounded-lg border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            title="Start new topic"
          >
            <Plus className="w-4 h-4 text-text-muted" />
            <span className="text-xs font-medium text-text-muted">New Topic</span>
          </button>
        </form>

        {!promptTitle ? (
          <p
            className={cn(
              "mt-2 text-xs text-text-muted text-center",
              useStickyMobileLayout && "mx-auto max-w-4xl",
            )}
          >
            {helperText}
          </p>
        ) : null}
        {showCharacterCount ? (
          <p
            className={cn(
              "mt-1 text-right text-xs",
              useStickyMobileLayout && "mx-auto max-w-4xl",
              hitCharacterLimit ? "text-warn" : "text-text-muted",
            )}
          >
            {input.length}/{MAX_FOLLOW_UP_LENGTH}
            {hitCharacterLimit
              ? " • Max length reached. Extra text will not be included."
              : ""}
          </p>
        ) : null}
      </div>
    </>
  )
}
