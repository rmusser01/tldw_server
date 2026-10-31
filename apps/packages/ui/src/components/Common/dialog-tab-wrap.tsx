import React from "react"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"

const TABBABLE_SELECTOR = [
  "a[href]",
  "button",
  "input:not([type='hidden'])",
  "select",
  "textarea",
  "[tabindex]",
  "[contenteditable='true']"
].join(", ")

/** The controls inside `container` that the Tab key can reach, in DOM order. */
export function tabbableWithin(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(TABBABLE_SELECTOR)
  ).filter(
    (element) =>
      element.tabIndex >= 0 &&
      !element.hasAttribute("disabled") &&
      !isHiddenBySelfOrAncestor(element)
  )
}

/**
 * Keeps Tab and Shift+Tab inside a dialog.
 *
 * antd 6 only pulls focus back once it lands on another element. A modal is
 * the last thing in the document, so Tab from its last control leaves the
 * page and focus sits on <body> for one stop (AX-09). This moves focus to
 * the other end of the dialog instead. Key presses a control has already
 * handled (for example Tab in an editor) are left alone.
 */
export function wrapTabAtDialogEdges(
  event: React.KeyboardEvent<HTMLElement>
): void {
  if (
    event.key !== "Tab" ||
    event.defaultPrevented ||
    event.altKey ||
    event.ctrlKey ||
    event.metaKey
  ) {
    return
  }
  const controls = tabbableWithin(event.currentTarget)
  if (controls.length === 0) return
  const first = controls[0]
  const last = controls[controls.length - 1]
  const active = document.activeElement

  if (!event.shiftKey && active === last) {
    event.preventDefault()
    first.focus()
  } else if (event.shiftKey && active === first) {
    event.preventDefault()
    last.focus()
  }
}

/**
 * For antd Modal's `modalRender`: wraps the dialog content so Tab stays
 * inside it. `display: contents` leaves the dialog layout unchanged.
 */
export function renderTabWrappedDialog(node: React.ReactNode): React.ReactNode {
  return (
    <div className="contents" onKeyDown={wrapTabAtDialogEdges}>
      {node}
    </div>
  )
}
