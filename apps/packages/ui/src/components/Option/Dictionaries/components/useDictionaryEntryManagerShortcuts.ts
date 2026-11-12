import React from "react"
import { isEditableTarget } from "@/utils/editable-target"

type UseDictionaryEntryManagerShortcutsParams = {
  editingEntry: any | null
  form: { submit: () => void }
  editEntryForm: { submit: () => void }
  runValidation: () => Promise<unknown> | unknown
  openValidationPanel: () => void
}

export function useDictionaryEntryManagerShortcuts({
  editingEntry,
  form,
  editEntryForm,
  runValidation,
  openValidationPanel,
}: UseDictionaryEntryManagerShortcutsParams): void {
  React.useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return

      const hasModifier = event.ctrlKey || event.metaKey
      if (!hasModifier || event.altKey) return

      const lowered = event.key.toLowerCase()

      if (lowered === "v" && event.shiftKey) {
        if (isEditableTarget(event.target)) return
        event.preventDefault()
        openValidationPanel()
        void runValidation()
        return
      }

      if (event.key !== "Enter" || event.shiftKey) return

      event.preventDefault()
      if (editingEntry) {
        editEntryForm.submit()
        return
      }
      form.submit()
    }

    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [editEntryForm, editingEntry, form, openValidationPanel, runValidation])
}
