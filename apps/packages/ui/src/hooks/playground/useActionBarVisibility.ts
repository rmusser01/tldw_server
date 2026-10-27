import React from "react"

export type UseActionBarVisibilityParams = {
  /**
   * Whether any external source (open modal, popover, dropdown, etc.)
   * should keep the action bar pinned open.  The hook will additionally
   * pin when the composer has focus-within.
   */
  externalPinSources: boolean
}

export function useActionBarVisibility({
  externalPinSources
}: UseActionBarVisibilityParams) {
  const [actionBarVisible, setActionBarVisible] = React.useState(true)
  const [composerHovering, setComposerHovering] = React.useState(false)
  const [composerFocusWithin, setComposerFocusWithin] = React.useState(false)
  const [composerTextareaExpanded, setComposerTextareaExpanded] = React.useState(false)

  const isPinned = composerFocusWithin || externalPinSources

  const actionBarCollapseTimerRef = React.useRef<number | null>(null)
  const actionBarPinnedRef = React.useRef(false)
  const composerHoveringRef = React.useRef(false)

  const clearActionBarCollapseTimer = React.useCallback(() => {
    if (actionBarCollapseTimerRef.current !== null) {
      window.clearTimeout(actionBarCollapseTimerRef.current)
      actionBarCollapseTimerRef.current = null
    }
  }, [])

  const showActionBar = React.useCallback(() => {
    clearActionBarCollapseTimer()
    setActionBarVisible(true)
  }, [clearActionBarCollapseTimer])

  const scheduleActionBarCollapse = React.useCallback(() => {
    clearActionBarCollapseTimer()
    if (actionBarPinnedRef.current) return
    actionBarCollapseTimerRef.current = window.setTimeout(() => {
      if (!actionBarPinnedRef.current && !composerHoveringRef.current) {
        setActionBarVisible(false)
      }
    }, 800)
  }, [clearActionBarCollapseTimer])

  // Sync refs
  React.useEffect(() => {
    actionBarPinnedRef.current = isPinned
  }, [isPinned])

  React.useEffect(() => {
    composerHoveringRef.current = composerHovering
  }, [composerHovering])

  // Auto-show/collapse based on pin & hover state
  React.useEffect(() => {
    if (isPinned) {
      showActionBar()
      return
    }
    if (!composerHovering) {
      scheduleActionBarCollapse()
    }
  }, [isPinned, composerHovering, scheduleActionBarCollapse, showActionBar])

  // Cleanup timer on unmount
  React.useEffect(() => clearActionBarCollapseTimer, [clearActionBarCollapseTimer])

  const handleComposerMouseEnter = React.useCallback(() => {
    setComposerHovering(true)
    showActionBar()
  }, [showActionBar])

  const handleComposerMouseLeave = React.useCallback(() => {
    setComposerHovering(false)
    scheduleActionBarCollapse()
  }, [scheduleActionBarCollapse])

  const handleComposerFocusCapture = React.useCallback((event: React.FocusEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLTextAreaElement && event.currentTarget.contains(event.target))
      setComposerTextareaExpanded(true)
    setComposerFocusWithin(true)
    showActionBar()
  }, [showActionBar])

  const handleComposerBlurCapture = React.useCallback(
    (event: React.FocusEvent<HTMLDivElement>) => {
      const next = event.relatedTarget as Node | null
      if (next && event.currentTarget.contains(next)) return
      setComposerFocusWithin(false)
      setComposerTextareaExpanded(false)
      scheduleActionBarCollapse()
    },
    [scheduleActionBarCollapse]
  )

  const actionBarVisibilityClass = actionBarVisible
    ? "max-h-[480px] opacity-100 visible"
    : "max-h-0 opacity-0 invisible pointer-events-none"

  return {
    actionBarVisible,
    composerFocusWithin,
    composerTextareaExpanded,
    actionBarVisibilityClass,
    handlers: {
      onMouseEnter: handleComposerMouseEnter,
      onMouseLeave: handleComposerMouseLeave,
      onFocusCapture: handleComposerFocusCapture,
      onBlurCapture: handleComposerBlurCapture
    }
  }
}
