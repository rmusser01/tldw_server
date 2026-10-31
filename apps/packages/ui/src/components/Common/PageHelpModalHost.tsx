import React, { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useHelpModal } from "@/store/tutorials"

const loadPageHelpModal = () => import("./PageHelpModal")
const reloadCurrentPage = () => window.location.reload()

type PageHelpModalHostProps = {
  loadModal?: () => Promise<{ PageHelpModal: React.ComponentType }>
  reloadPage?: () => void
}

/** Keep help open events available without loading an optional chunk at startup. */
export function PageHelpModalHost({ loadModal = loadPageHelpModal, reloadPage = reloadCurrentPage }: PageHelpModalHostProps) {
  const { isOpen, open, close } = useHelpModal()
  const { t } = useTranslation("common")
  const [HelpModal, setHelpModal] = useState<React.ComponentType | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    window.addEventListener("tldw:open-help-modal", open)
    window.addEventListener("tldw:open-shortcuts-modal", open)
    return () => {
      window.removeEventListener("tldw:open-help-modal", open)
      window.removeEventListener("tldw:open-shortcuts-modal", open)
    }
  }, [open])

  useEffect(() => {
    if (!isOpen || HelpModal) return
    let cancelled = false
    setFailed(false)
    loadModal().then(
      (module) => { if (!cancelled) setHelpModal(() => module.PageHelpModal) },
      () => { if (!cancelled) setFailed(true) }
    )
    return () => { cancelled = true }
  }, [isOpen, HelpModal, loadModal])

  useEffect(() => {
    if (!isOpen) return
    const previousFocus = document.activeElement
    const onEscape = (event: KeyboardEvent) => { if (event.key === "Escape") close() }
    document.addEventListener("keydown", onEscape)
    return () => {
      document.removeEventListener("keydown", onEscape)
      if (previousFocus instanceof HTMLElement && document.contains(previousFocus)) previousFocus.focus()
    }
  }, [isOpen, close])

  if (!isOpen) return null
  if (HelpModal) return <HelpModal />

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-sm rounded border border-border bg-surface p-4 shadow-lg" role={failed ? "alert" : "status"}>
      <p className="text-sm text-text">
        {failed
          ? t("help.loadFailed", "Help could not be loaded. Reconnect, save any edits, then reload the page.")
          : t("help.loading", "Loading help…")}
      </p>
      <div className="mt-2 flex gap-3 text-sm">
        {failed && (
          <button type="button" className="text-primary" onClick={() => {
            if (window.confirm(t("help.confirmReload", "Reload this page? Unsaved edits may be lost."))) reloadPage()
          }}>
            {t("help.reload", "Reload page")}
          </button>
        )}
        <button type="button" className="text-text-muted" onClick={close}>{t("dismiss", "Dismiss")}</button>
      </div>
    </div>
  )
}
