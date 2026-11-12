import React from "react"
import { isEditableTarget } from "@/utils/editable-target"

type UseDictionaryManagerShortcutsParams = {
  openCreate: boolean
  openEdit: boolean
  openCreateDictionaryModal: () => void
  createForm: { submit: () => void }
  editForm: { submit: () => void }
}

export function useDictionaryManagerShortcuts({
  openCreate,
  openEdit,
  openCreateDictionaryModal,
  createForm,
  editForm,
}: UseDictionaryManagerShortcutsParams): void {
  React.useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return

      const hasModifier = event.ctrlKey || event.metaKey
      if (!hasModifier || event.altKey) return

      const lowered = event.key.toLowerCase()

      if (lowered === "n" && !event.shiftKey) {
        if (isEditableTarget(event.target)) return
        if (openCreate || openEdit) return
        event.preventDefault()
        openCreateDictionaryModal()
        return
      }

      if (event.key !== "Enter" || event.shiftKey) return

      if (openEdit) {
        event.preventDefault()
        editForm.submit()
        return
      }

      if (openCreate) {
        event.preventDefault()
        createForm.submit()
      }
    }

    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [
    createForm,
    editForm,
    openCreate,
    openCreateDictionaryModal,
    openEdit,
  ])
}
