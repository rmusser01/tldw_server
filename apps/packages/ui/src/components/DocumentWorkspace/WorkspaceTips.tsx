import React, { useState, useEffect, useCallback, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Tour } from "antd"
import type { TourProps } from "antd"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

const TOUR_KEY = "dw-tips-tour-completed"
const TIP_HIGHLIGHT_KEY = "dw-tips-highlight-shown"
const TIP_MULTIDOC_KEY = "dw-tips-multidoc-shown"

const getFlag = (key: string): boolean => {
  try { return localStorage.getItem(key) === "true" } catch { return false }
}

const setFlag = (key: string) => {
  try { localStorage.setItem(key, "true") } catch { /* noop */ }
}

export const resetAllTips = () => {
  try {
    localStorage.removeItem(TOUR_KEY)
    localStorage.removeItem(TIP_HIGHLIGHT_KEY)
    localStorage.removeItem(TIP_MULTIDOC_KEY)
  } catch { /* noop */ }
}

export const resetTour = () => {
  try { localStorage.removeItem(TOUR_KEY) } catch { /* noop */ }
}

/**
 * WorkspaceTour - Ant Design Tour that runs on first visit.
 *
 * Targets header buttons and panels using data-testid selectors.
 * Automatically skips if already completed.
 */
export const WorkspaceTour: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!getFlag(TOUR_KEY)) {
      // Delay so DOM elements are mounted
      const timer = setTimeout(() => setOpen(true), 500)
      return () => clearTimeout(timer)
    }
  }, [])

  const handleClose = useCallback(() => {
    setOpen(false)
    setFlag(TOUR_KEY)
  }, [])

  const steps: TourProps["steps"] = [
    {
      title: t("option:documentWorkspace.tourWelcome", "Welcome to Document Workspace"),
      description: t(
        "option:documentWorkspace.tourWelcomeDesc",
        "Read, annotate, and interact with your PDF and EPUB documents. Let's take a quick look around."
      ),
      target: null
    },
    {
      title: t("option:documentWorkspace.tourOpen", "Open a document"),
      description: t(
        "option:documentWorkspace.tourOpenDesc",
        "Click here to open a document from your library or upload a new one."
      ),
      target: () => document.querySelector<HTMLElement>('[aria-label="Open document"]')
    },
    {
      title: t("option:documentWorkspace.tourSidebar", "Explore your tools"),
      description: t(
        "option:documentWorkspace.tourSidebarDesc",
        "The sidebar has tabs for table of contents, AI insights, document info, and more. Use the \"More\" button to see additional tabs."
      ),
      target: () => document.querySelector<HTMLElement>('[data-testid="document-workspace-toggle-left"]'),
      placement: "right"
    },
    {
      title: t("option:documentWorkspace.tourChat", "Chat with your document"),
      description: t(
        "option:documentWorkspace.tourChatDesc",
        "Ask AI questions and get answers based on your document's content. Enable \"Use document content\" for the best results."
      ),
      target: () => document.querySelector<HTMLElement>('[data-testid="document-workspace-toggle-right"]'),
      placement: "left"
    },
    {
      title: t("option:documentWorkspace.tourHelp", "Help menu"),
      description: t(
        "option:documentWorkspace.tourHelpDesc",
        "Find keyboard shortcuts, feature tips, and this tour again in the Help menu. Press ? anytime for shortcuts."
      ),
      target: () => document.querySelector<HTMLElement>('[aria-label="Help"]')
    }
  ]

  if (!open) return null

  return (
    <Tour
      open={open}
      onClose={handleClose}
      steps={steps}
      type="primary"
    />
  )
}

/**
 * HighlightTip - One-time transient tip shown on first document open.
 *
 * Reminds users they can select text to highlight or ask AI about it.
 */
export const HighlightTip: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (activeDocumentId && !getFlag(TIP_HIGHLIGHT_KEY)) {
      const timer = setTimeout(() => setVisible(true), 1500)
      return () => clearTimeout(timer)
    }
  }, [activeDocumentId])

  useEffect(() => {
    if (visible) {
      const timer = setTimeout(() => {
        setVisible(false)
        setFlag(TIP_HIGHLIGHT_KEY)
      }, 8000)
      return () => clearTimeout(timer)
    }
  }, [visible])

  const handleDismiss = useCallback(() => {
    setVisible(false)
    setFlag(TIP_HIGHLIGHT_KEY)
  }, [])

  if (!visible) return null

  return (
    <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 shadow-md animate-in fade-in slide-in-from-top-2">
      <span className="text-xs text-text-muted">
        {t(
          "option:documentWorkspace.tipHighlight",
          "Tip: Select text to highlight or ask AI about it"
        )}
      </span>
      <button
        onClick={handleDismiss}
        className="text-xs text-text-subtle hover:text-text ml-1"
        aria-label={t("common:dismiss", "Dismiss")}
      >
        &times;
      </button>
    </div>
  )
}

/**
 * MultiDocTip - One-time tip shown when tab bar first appears.
 */
export const MultiDocTip: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const openDocuments = useDocumentWorkspaceStore((s) => s.openDocuments)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (openDocuments.length === 1 && !getFlag(TIP_MULTIDOC_KEY)) {
      const timer = setTimeout(() => setVisible(true), 2000)
      return () => clearTimeout(timer)
    }
  }, [openDocuments.length])

  useEffect(() => {
    if (visible) {
      const timer = setTimeout(() => {
        setVisible(false)
        setFlag(TIP_MULTIDOC_KEY)
      }, 6000)
      return () => clearTimeout(timer)
    }
  }, [visible])

  const handleDismiss = useCallback(() => {
    setVisible(false)
    setFlag(TIP_MULTIDOC_KEY)
  }, [])

  if (!visible) return null

  return (
    <div className="absolute top-0 right-12 z-10 flex items-center gap-2 rounded-b-lg border border-t-0 border-border bg-surface px-3 py-1.5 shadow-sm">
      <span className="text-xs text-text-muted">
        {t(
          "option:documentWorkspace.tipMultiDoc",
          "You can open multiple documents and switch between tabs"
        )}
      </span>
      <button
        onClick={handleDismiss}
        className="text-xs text-text-subtle hover:text-text ml-1"
        aria-label={t("common:dismiss", "Dismiss")}
      >
        &times;
      </button>
    </div>
  )
}

export default WorkspaceTour
