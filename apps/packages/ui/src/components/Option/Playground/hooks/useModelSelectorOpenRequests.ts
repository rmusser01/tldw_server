import React from "react"
import { useShortcut } from "@/hooks/useKeyboardShortcuts"
import {
  focusFirstVisibleElement,
  normalizeFocusSelector,
} from "@/utils/focus-return"
import {
  OPEN_MODEL_SELECTOR_EVENT,
  type ModelSelectorOpenDetail,
} from "../playground-cockpit-actions"

const MODEL_SELECTOR = "[data-testid='model-selector']"

type ModelSelectorOpenRequestsOptions = {
  /** Opens the composer's model picker. */
  openModelApiSelector: () => void
  /** Where focus goes when a picker opened by a request closes. */
  returnFocusSelectorRef: React.MutableRefObject<string | null>
}

/**
 * Open the composer's model picker when another part of the page asks for it
 * with `openModelSelector()` (the command palette's Switch Model, `/model`,
 * chat recovery), or when the user presses the Switch model shortcut, Mod+E
 * (AX-01). Taking a request cancels its event, which tells the palette not to
 * fall back to the chat settings.
 */
export const useModelSelectorOpenRequests = ({
  openModelApiSelector,
  returnFocusSelectorRef,
}: ModelSelectorOpenRequestsOptions) => {
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

  useShortcut(
    {
      key: "e",
      modifiers: ["meta"],
      description: "Switch model",
      allowInInput: true,
      action: () => openFromRequest(),
    },
    [openFromRequest],
  )
}
