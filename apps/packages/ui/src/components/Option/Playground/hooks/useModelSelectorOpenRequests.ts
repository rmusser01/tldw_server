import React from "react"
import { matchesShortcut, type Shortcut } from "@/hooks/useKeyboardShortcuts"
import {
  focusFirstVisibleElement,
  normalizeFocusSelector,
} from "@/utils/focus-return"
import {
  OPEN_MODEL_SELECTOR_EVENT,
  type ModelSelectorOpenDetail,
} from "../playground-cockpit-actions"

const MODEL_SELECTOR = "[data-testid='model-selector']"
const DIALOG = "[aria-modal='true'], [role='dialog']"
/** The Switch model shortcut: Cmd+E on macOS, Ctrl+E elsewhere. */
const SWITCH_MODEL_SHORTCUT: Pick<Shortcut, "key" | "modifiers"> = {
  key: "e",
  modifiers: ["meta"],
}

type ModelSelectorOpenRequestsOptions = {
  /** Opens the composer's model picker. */
  openModelApiSelector: () => void
  /** Where focus goes when a picker opened by a request closes. */
  returnFocusSelectorRef: React.MutableRefObject<string | null>
  /** Whether the picker is open now. */
  pickerOpen: boolean
}

/** Whether `node` sits in a dialog that does not hold the composer. */
const inOtherDialog = (node: EventTarget | Element | null) => {
  if (!(node instanceof Element)) return false
  const dialog = node.closest(DIALOG)
  return Boolean(dialog && !dialog.querySelector(MODEL_SELECTOR))
}

/**
 * Open the composer's model picker when another part of the page asks for it
 * with `openModelSelector()` (the command palette's Switch Model, `/model`,
 * chat recovery), or when the user presses the Switch model shortcut, Mod+E
 * (AX-01). Taking a request cancels its event, which tells the palette not to
 * fall back to the chat settings.
 *
 * Mod+E works while typing in the composer, but not from inside another
 * dialog (a modal's focus lock would trap the picker), not while the key
 * repeats, and not when the picker is already open, so focus stays in its
 * search.
 */
export const useModelSelectorOpenRequests = ({
  openModelApiSelector,
  returnFocusSelectorRef,
  pickerOpen,
}: ModelSelectorOpenRequestsOptions) => {
  const pickerOpenRef = React.useRef(pickerOpen)
  pickerOpenRef.current = pickerOpen

  const openFromRequest = React.useCallback(
    (returnFocusSelector?: string) => {
      focusFirstVisibleElement(MODEL_SELECTOR)
      returnFocusSelectorRef.current = normalizeFocusSelector(returnFocusSelector)
      openModelApiSelector()
    },
    [openModelApiSelector, returnFocusSelectorRef],
  )

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handler = (event: Event) => {
      event.preventDefault()
      const detail = (event as CustomEvent<ModelSelectorOpenDetail>).detail
      openFromRequest(detail?.returnFocusSelector)
    }
    window.addEventListener(OPEN_MODEL_SELECTOR_EVENT, handler)
    return () => {
      window.removeEventListener(OPEN_MODEL_SELECTOR_EVENT, handler)
    }
  }, [openFromRequest])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const onKeyDown = (event: KeyboardEvent) => {
      if (!matchesShortcut(event, SWITCH_MODEL_SHORTCUT)) return
      if (inOtherDialog(event.target) || inOtherDialog(document.activeElement)) return
      event.preventDefault()
      event.stopPropagation()
      if (event.repeat || pickerOpenRef.current) return
      openFromRequest()
    }
    window.addEventListener("keydown", onKeyDown, { capture: true })
    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true })
    }
  }, [openFromRequest])
}
