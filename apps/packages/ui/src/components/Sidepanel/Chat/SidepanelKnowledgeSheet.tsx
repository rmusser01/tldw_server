/**
 * Search & Context in the side panel, as an overlay sheet (XS-15).
 *
 * The shared KnowledgePanel used to open inline above the composer and push
 * the textarea about 1,200px down. Here it opens as a modal bottom sheet over
 * the conversation, with its own height cap. It takes focus as it opens, so
 * Escape works at once; Escape, its Close button or a click outside close
 * it, and focus goes back to the control that opened it. The panel stays
 * mounted while closed because it listens for the open event itself
 * ("tldw:toggle-rag"). The sheet is portalled to <body> so the composer
 * dock's overflow cannot clip it.
 */
import React from "react"
import { createPortal } from "react-dom"
import { useTranslation } from "react-i18next"
import { KnowledgePanel, type KnowledgePanelProps } from "@/components/Knowledge"

export const SidepanelKnowledgeSheet: React.FC<KnowledgePanelProps> = (props) => {
  const { t } = useTranslation(["playground"])
  const open = Boolean(props.open)
  const { onOpenChange } = props
  const sheetRef = React.useRef<HTMLDivElement>(null)
  const returnFocusRef = React.useRef<HTMLElement | null>(null)

  React.useLayoutEffect(() => {
    if (open) {
      const active = document.activeElement
      returnFocusRef.current =
        active instanceof HTMLElement && !sheetRef.current?.contains(active) ? active : null
      const frame = requestAnimationFrame(() => {
        const sheet = sheetRef.current
        if (sheet && !sheet.contains(document.activeElement)) sheet.focus()
      })
      return () => cancelAnimationFrame(frame)
    }
    const target = returnFocusRef.current
    returnFocusRef.current = null
    if (target?.isConnected) target.focus()
    return undefined
  }, [open])

  const sheet = (
    <>
      {open && (
        <div
          data-testid="sidepanel-knowledge-sheet-backdrop"
          aria-hidden="true"
          onMouseDown={() => onOpenChange?.(false)}
          className="fixed inset-0 z-40 bg-black/20"
        />
      )}
      <div
        ref={sheetRef}
        data-testid="sidepanel-knowledge-sheet"
        role={open ? "dialog" : undefined}
        aria-modal={open ? true : undefined}
        aria-label={
          open ? t("playground:composer.contextKnowledge", "Search & Context") : undefined
        }
        tabIndex={-1}
        hidden={!open}
        onKeyDown={(event) => {
          if (event.key !== "Escape" || event.defaultPrevented) return
          event.stopPropagation()
          onOpenChange?.(false)
        }}
        className="fixed inset-x-0 bottom-0 z-50 max-h-[75vh] overflow-y-auto overscroll-contain rounded-t-xl border-t border-border bg-surface px-2 pt-2 shadow-2xl focus:outline-none"
      >
        <KnowledgePanel {...props} />
      </div>
    </>
  )
  if (typeof document === "undefined") return sheet
  return createPortal(sheet, document.body)
}
