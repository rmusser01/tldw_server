import React, { useEffect, useLayoutEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useLocation } from "react-router-dom"
import { CHAT_PATH } from "@/routes/route-paths"
import { useHelpModal } from "@/store/tutorials"
import { shouldOpenShortcutsHelp } from "@/components/Option/Playground/playground-shortcuts"
import { isEditableTarget } from "@/utils/editable-target"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"
import type { PageHelpModalProps } from "./PageHelpModal"

const loadPageHelpModal = () => import("./PageHelpModal")
const reloadCurrentPage = () => window.location.reload()
const hasOtherOpenDialog = () =>
  Array.from(
    document.querySelectorAll<HTMLElement>(
      ':is([role="dialog"], [role="alertdialog"])[aria-modal="true"]:not([aria-labelledby="help-modal-title"])'
    )
  ).some((dialog) => !isHiddenBySelfOrAncestor(dialog))

type PageHelpModalHostProps = Pick<
  PageHelpModalProps,
  "scope" | "modeNavigationEnabled"
> & {
  loadModal?: () => Promise<{
    PageHelpModal: React.ComponentType<PageHelpModalProps>
  }>
  reloadPage?: () => void
}

/** Keep help open events available without loading an optional chunk at startup. */
export function PageHelpModalHost({
  loadModal = loadPageHelpModal,
  reloadPage = reloadCurrentPage,
  scope = "full",
  modeNavigationEnabled = true
}: PageHelpModalHostProps) {
  const { isOpen, open, close } = useHelpModal()
  const { pathname } = useLocation()
  const chatKeyboardEnabled =
    pathname === CHAT_PATH || (scope === "sidepanel" && pathname === "/")
  const { t } = useTranslation("common")
  const [HelpModal, setHelpModal] =
    useState<React.ComponentType<PageHelpModalProps> | null>(null)
  const [failed, setFailed] = useState(false)
  const [initialTab, setInitialTab] =
    useState<PageHelpModalProps["initialTab"]>()

  useEffect(() => {
    const openHelp = () => {
      setInitialTab(undefined)
      open()
    }
    const openShortcuts = () => {
      setInitialTab("shortcuts")
      open()
    }
    const onQuestionMark = (event: KeyboardEvent) => {
      if (
        !chatKeyboardEnabled ||
        event.defaultPrevented ||
        event.repeat ||
        !shouldOpenShortcutsHelp(event) ||
        isEditableTarget(document.activeElement)
      )
        return
      if (hasOtherOpenDialog()) return
      event.preventDefault()
      if (isOpen) close()
      else openShortcuts()
    }
    window.addEventListener("tldw:open-help-modal", openHelp)
    window.addEventListener("tldw:open-shortcuts-modal", openShortcuts)
    window.addEventListener("tldw:open-playground-shortcuts", openShortcuts)
    window.addEventListener("keydown", onQuestionMark)
    return () => {
      window.removeEventListener("tldw:open-help-modal", openHelp)
      window.removeEventListener("tldw:open-shortcuts-modal", openShortcuts)
      window.removeEventListener(
        "tldw:open-playground-shortcuts",
        openShortcuts
      )
      window.removeEventListener("keydown", onQuestionMark)
    }
  }, [open, close, isOpen, chatKeyboardEnabled])

  useEffect(() => {
    if (!isOpen) setInitialTab(undefined)
  }, [isOpen])

  useEffect(() => {
    if (!isOpen || HelpModal) return
    let cancelled = false
    setFailed(false)
    loadModal().then(
      (module) => {
        if (!cancelled) setHelpModal(() => module.PageHelpModal)
      },
      () => {
        if (!cancelled) setFailed(true)
      }
    )
    return () => {
      cancelled = true
    }
  }, [isOpen, HelpModal, loadModal])

  // Capture the initiator before a cached modal focuses its close button.
  useLayoutEffect(() => {
    if (!isOpen) return
    const previousFocus = document.activeElement
    const onEscape = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.key !== "Escape") return
      if (hasOtherOpenDialog()) return
      event.preventDefault()
      event.stopPropagation()
      close()
    }
    window.addEventListener("keydown", onEscape, true)
    return () => {
      window.removeEventListener("keydown", onEscape, true)
      if (
        previousFocus instanceof HTMLElement &&
        document.contains(previousFocus)
      )
        previousFocus.focus()
    }
  }, [isOpen, close])

  if (!isOpen) return null
  if (HelpModal)
    return (
      <HelpModal
        initialTab={initialTab}
        scope={scope}
        modeNavigationEnabled={modeNavigationEnabled}
      />
    )

  return (
    <div
      className="fixed bottom-4 right-4 z-50 max-w-sm rounded border border-border bg-surface p-4 shadow-lg"
      role={failed ? "alert" : "status"}>
      <p className="text-sm text-text">
        {failed
          ? t(
              "help.loadFailed",
              "Help could not be loaded. Reconnect, save any edits, then reload the page."
            )
          : t("help.loading", "Loading help…")}
      </p>
      <div className="mt-2 flex gap-3 text-sm">
        {failed && (
          <button
            type="button"
            className="text-primary"
            onClick={() => {
              if (
                window.confirm(
                  t(
                    "help.confirmReload",
                    "Reload this page? Unsaved edits may be lost."
                  )
                )
              )
                reloadPage()
            }}>
            {t("help.reload", "Reload page")}
          </button>
        )}
        <button type="button" className="text-text-muted" onClick={close}>
          {t("dismiss", "Dismiss")}
        </button>
      </div>
    </div>
  )
}
