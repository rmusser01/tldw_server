import React, { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { createPortal } from "react-dom"
import { X, Keyboard } from "lucide-react"

interface ShortcutGroup {
  title: string
  shortcuts: {
    label: string
    keys: string
  }[]
}

const isMac =
  typeof navigator !== "undefined" &&
  /Mac|iPod|iPhone|iPad/.test(navigator.platform)

interface DocumentShortcutsModalProps {
  open: boolean
  onClose: () => void
}

/**
 * Modal displaying keyboard shortcuts available in the Document Workspace.
 * Can be triggered by pressing ? (question mark) while viewing a document.
 */
export function DocumentShortcutsModal({
  open,
  onClose
}: DocumentShortcutsModalProps) {
  const { t } = useTranslation(["option", "common"])
  const modalRef = useRef<HTMLDivElement | null>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const previouslyFocusedElementRef = useRef<HTMLElement | null>(null)

  // Focus management
  useEffect(() => {
    if (open) {
      previouslyFocusedElementRef.current =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null

      if (closeButtonRef.current) {
        closeButtonRef.current.focus()
      }
      return
    }

    if (!open && previouslyFocusedElementRef.current) {
      const element = previouslyFocusedElementRef.current
      previouslyFocusedElementRef.current = null

      if (document.contains(element)) {
        element.focus()
      }
    }
  }, [open])

  // Close on Escape
  useEffect(() => {
    if (!open) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault()
        onClose()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [open, onClose])

  const modKey = isMac ? "⌘" : "Ctrl"

  const shortcutGroups: ShortcutGroup[] = useMemo(
    () => [
      {
        title: "Navigation",
        shortcuts: [
          { label: "Next page", keys: "→ or Page Down" },
          { label: "Previous page", keys: "← or Page Up" },
          { label: "First page (PDF)", keys: "Home" },
          { label: "Last page (PDF)", keys: "End" },
          { label: "Go to page", keys: `${modKey} + G` }
        ]
      },
      {
        title: "View",
        shortcuts: [
          { label: "Zoom in", keys: `${modKey} + +` },
          { label: "Zoom out", keys: `${modKey} + -` },
          { label: "Reset zoom", keys: `${modKey} + 0` },
          { label: "Toggle fullscreen", keys: "F" }
        ]
      },
      {
        title: "Search",
        shortcuts: [
          { label: "Find in document", keys: `${modKey} + F` },
          { label: "Next result", keys: "Enter" },
          { label: "Previous result", keys: "Shift + Enter" },
          { label: "Close search", keys: "Escape" }
        ]
      },
      {
        title: "Panels",
        shortcuts: [
          { label: "Toggle left sidebar", keys: `${modKey} + [` },
          { label: "Toggle right panel", keys: `${modKey} + ]` },
          { label: "Focus chat input", keys: `${modKey} + /` }
        ]
      },
      {
        title: "General",
        shortcuts: [
          { label: "Show shortcuts", keys: "?" }
        ]
      }
    ],
    [modKey]
  )

  const handleKeyDownInModal = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== "Tab" || !modalRef.current) {
        return
      }

      const focusableSelectors =
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])'
      const focusableElements = Array.from(
        modalRef.current.querySelectorAll<HTMLElement>(focusableSelectors)
      ).filter(
        (element) =>
          !element.hasAttribute("disabled") &&
          element.getAttribute("aria-hidden") !== "true" &&
          element.tabIndex !== -1
      )

      if (focusableElements.length === 0) {
        event.preventDefault()
        return
      }

      const currentIndex = focusableElements.indexOf(
        document.activeElement as HTMLElement
      )
      let nextIndex = currentIndex

      if (event.shiftKey) {
        nextIndex =
          currentIndex <= 0 ? focusableElements.length - 1 : currentIndex - 1
      } else {
        nextIndex =
          currentIndex === -1 || currentIndex === focusableElements.length - 1
            ? 0
            : currentIndex + 1
      }

      focusableElements[nextIndex].focus()
      event.preventDefault()
    },
    []
  )

  if (!open) return null
  if (typeof document === "undefined") return null

  const modalContent = (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-bg/70 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal */}
      <div
        ref={modalRef}
        className="fixed left-1/2 top-[10%] sm:top-[15%] z-50 w-[calc(100%-2rem)] sm:w-full max-w-lg -translate-x-1/2 overflow-hidden rounded-xl border border-border bg-surface shadow-2xl max-h-[80vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-labelledby="doc-shortcuts-modal-title"
        onKeyDown={handleKeyDownInModal}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <Keyboard className="size-5 text-text-subtle" />
            <h2
              id="doc-shortcuts-modal-title"
              className="text-base font-semibold text-text"
            >
              {t("option:documentWorkspace.shortcutsTitle", "Document Workspace Shortcuts")}
            </h2>
          </div>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="rounded-lg p-2 min-w-[44px] min-h-[44px] flex items-center justify-center text-text-subtle hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        {/* Content */}
        <div className="max-h-[60vh] overflow-y-auto p-4">
          {shortcutGroups.map((group, groupIndex) => (
            <div key={group.title} className={groupIndex > 0 ? "mt-5" : ""}>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-subtle">
                {group.title}
              </h3>
              <div className="space-y-1">
                {group.shortcuts.map((shortcut) => (
                  <div
                    key={shortcut.label}
                    className="flex items-center justify-between rounded-lg px-2 py-1.5 hover:bg-surface2"
                  >
                    <span className="text-sm text-text">{shortcut.label}</span>
                    <kbd className="ml-4 flex items-center gap-1 rounded border border-border bg-surface2 px-2 py-0.5 text-xs font-medium text-text-muted">
                      {shortcut.keys}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-center border-t border-border px-4 py-2.5">
          <p className="text-xs text-text-subtle">
            Press <kbd className="mx-1 rounded border border-border bg-surface2 px-1.5 py-0.5 text-xs">?</kbd> anytime to show this help
          </p>
        </div>
      </div>
    </>
  )

  return createPortal(modalContent, document.body)
}

export default DocumentShortcutsModal
