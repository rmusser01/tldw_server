/**
 * SettingsPanel - RAG settings drawer
 */

import React, { useEffect, useRef, useCallback } from "react"
import { X, Settings, Zap, Scale, Brain, Beaker, RotateCcw } from "lucide-react"
import { useKnowledgeQA } from "../KnowledgeQAProvider"
import { PresetSelector } from "./PresetSelector"
import { BasicSettings } from "./BasicSettings"
import { ExpertSettings } from "./ExpertSettings"
import { cn } from "@/libs/utils"

type SettingsPanelProps = {
  open: boolean
  onClose: () => void
  className?: string
}

// Get all focusable elements within a container
function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const focusableSelectors = [
    'button:not([disabled])',
    '[href]',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',')

  return Array.from(container.querySelectorAll(focusableSelectors)) as HTMLElement[]
}

// Local storage key for expert mode onboarding
const EXPERT_MODE_SEEN_KEY = 'knowledgeqa-expert-mode-seen'

function safeGetItem(key: string): string | null {
  if (typeof window === 'undefined') return null
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function safeSetItem(key: string, value: string): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(key, value)
  } catch {
    // Storage may be unavailable in private or restricted contexts.
  }
}

export function SettingsPanel({ open, onClose, className }: SettingsPanelProps) {
  const { expertMode, toggleExpertMode, resetSettings, preset } = useKnowledgeQA()
  const panelRef = useRef<HTMLDivElement>(null)
  const previousActiveElement = useRef<HTMLElement | null>(null)
  const expertModeHintTimeoutRef = useRef<number | null>(null)
  const [showExpertModeHint, setShowExpertModeHint] = React.useState(false)

  // Check if user has seen Expert Mode onboarding
  const hasSeenExpertMode = useCallback(() => {
    return safeGetItem(EXPERT_MODE_SEEN_KEY) === 'true'
  }, [])

  // Mark Expert Mode as seen
  const markExpertModeSeen = useCallback(() => {
    safeSetItem(EXPERT_MODE_SEEN_KEY, 'true')
  }, [])

  const clearExpertModeHintTimeout = useCallback(() => {
    if (expertModeHintTimeoutRef.current !== null) {
      window.clearTimeout(expertModeHintTimeoutRef.current)
      expertModeHintTimeoutRef.current = null
    }
  }, [])

  // Handle Expert Mode toggle with onboarding
  const handleExpertModeToggle = useCallback(() => {
    const isEnteringExpertMode = !expertMode
    toggleExpertMode()

    if (!isEnteringExpertMode) {
      clearExpertModeHintTimeout()
      setShowExpertModeHint(false)
      return
    }

    if (isEnteringExpertMode && !hasSeenExpertMode()) {
      clearExpertModeHintTimeout()
      setShowExpertModeHint(true)
      markExpertModeSeen()
      // Auto-dismiss after 6 seconds
      expertModeHintTimeoutRef.current = window.setTimeout(() => {
        setShowExpertModeHint(false)
        expertModeHintTimeoutRef.current = null
      }, 6000)
    }
  }, [clearExpertModeHintTimeout, expertMode, toggleExpertMode, hasSeenExpertMode, markExpertModeSeen])

  // Store the previously focused element when panel opens
  useEffect(() => {
    if (open) {
      previousActiveElement.current = document.activeElement as HTMLElement
    }
  }, [open])

  useEffect(() => {
    if (!open) {
      clearExpertModeHintTimeout()
      setShowExpertModeHint(false)
    }
  }, [clearExpertModeHintTimeout, open])

  useEffect(() => {
    if (!expertMode) {
      clearExpertModeHintTimeout()
      setShowExpertModeHint(false)
    }
  }, [clearExpertModeHintTimeout, expertMode])

  useEffect(() => {
    return () => {
      clearExpertModeHintTimeout()
    }
  }, [clearExpertModeHintTimeout])

  // Focus trap and keyboard handling
  useEffect(() => {
    if (!open || !panelRef.current) return

    const panel = panelRef.current

    // Focus the first focusable element
    const focusableElements = getFocusableElements(panel)
    if (focusableElements.length > 0) {
      focusableElements[0].focus()
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      // Close on Escape
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        onClose()
        return
      }

      // Focus trap on Tab
      if (e.key === 'Tab') {
        const focusableElements = getFocusableElements(panel)
        if (focusableElements.length === 0) return

        const firstElement = focusableElements[0]
        const lastElement = focusableElements[focusableElements.length - 1]

        if (e.shiftKey) {
          // Shift+Tab: go to last element if at first
          if (document.activeElement === firstElement) {
            e.preventDefault()
            lastElement.focus()
          }
        } else {
          // Tab: go to first element if at last
          if (document.activeElement === lastElement) {
            e.preventDefault()
            firstElement.focus()
          }
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown, true)
    return () => document.removeEventListener('keydown', handleKeyDown, true)
  }, [open, onClose])

  // Restore focus when closing
  useEffect(() => {
    if (!open && previousActiveElement.current) {
      previousActiveElement.current.focus()
      previousActiveElement.current = null
    }
  }, [open])

  if (!open) {
    return null
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-panel-title"
        className={cn(
          "fixed right-0 top-0 h-full w-96 max-w-[calc(100vw-2rem)]",
          "bg-surface border-l border-border shadow-xl",
          "flex flex-col z-50",
          "animate-in slide-in-from-right duration-200",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-text-muted" />
            <span id="settings-panel-title" className="font-semibold">RAG Settings</span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close settings panel"
            className="p-1.5 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {/* Preset selector */}
          <div className="p-4 border-b border-border">
            <PresetSelector />
          </div>

          {/* Mode toggle */}
          <div className="p-4 border-b border-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {expertMode ? (
                  <Beaker className="w-4 h-4 text-primary" />
                ) : (
                  <Zap className="w-4 h-4 text-text-muted" />
                )}
                <span id="expert-mode-label" className="text-sm font-medium">
                  {expertMode ? "Expert Mode" : "Basic Mode"}
                </span>
              </div>
              <button
                role="switch"
                aria-checked={expertMode}
                aria-labelledby="expert-mode-label"
                onClick={handleExpertModeToggle}
                className={cn(
                  "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                  expertMode ? "bg-primary" : "bg-muted"
                )}
              >
                <span
                  className={cn(
                    "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                    expertMode ? "translate-x-6" : "translate-x-1"
                  )}
                />
              </button>
            </div>
            <p className="text-xs text-text-muted mt-2">
              {expertMode
                ? "Full access to 150+ RAG options, including a complete key-level editor"
                : "Common options for quick configuration"}
            </p>

            {/* Expert Mode onboarding hint */}
            {showExpertModeHint && (
              <div className="mt-3 p-3 bg-primary/10 border border-primary/20 rounded-lg animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="flex items-start gap-2">
                  <Beaker className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-primary">Welcome to Expert Mode</p>
                    <p className="text-xs text-text-muted mt-1">
                      You now have access to advanced RAG controls by section, plus an All Options
                      editor for complete key-level access.
                    </p>
                    <button
                      onClick={() => setShowExpertModeHint(false)}
                      className="text-xs text-primary hover:underline mt-2"
                    >
                      Got it
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Settings content */}
          <div className="p-4">
            {expertMode ? <ExpertSettings /> : <BasicSettings />}
            <p className="mt-4 text-xs text-text-muted">
              Changes apply to your next search. Previous answers are not affected.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-border">
          <button
            onClick={resetSettings}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md hover:bg-muted transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset to Balanced Defaults
          </button>
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-sm font-medium rounded-md bg-primary text-white hover:bg-primaryStrong transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </>
  )
}
