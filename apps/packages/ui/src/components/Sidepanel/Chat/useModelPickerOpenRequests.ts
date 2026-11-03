import React from "react"

import { OPEN_MODEL_SELECTOR_EVENT } from "@/components/Option/Playground/playground-cockpit-actions"

type OpenablePicker = { openAndFocus: () => void }

/**
 * Open the side panel's model picker when another part of the page asks for
 * it with `openModelSelector()`: for example the history review's "Switch
 * model" after a send failed because of its model (CC-02).
 */
export const useModelPickerOpenRequests = (
  pickerRef: React.RefObject<OpenablePicker | null>
) => {
  React.useEffect(() => {
    const open = () => pickerRef.current?.openAndFocus()
    window.addEventListener(OPEN_MODEL_SELECTOR_EVENT, open)
    return () => window.removeEventListener(OPEN_MODEL_SELECTOR_EVENT, open)
  }, [pickerRef])
}
