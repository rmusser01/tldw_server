import React from "react"
import type { InputRef } from "antd"
import { useMutation, type QueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { useAntdNotification } from "@/hooks/useAntdNotification"

export interface UseCharacterInlineEditDeps {
  /** i18n translator */
  t: (key: string, opts?: Record<string, any>) => string
  /** Notification API */
  notification: ReturnType<typeof useAntdNotification>
  /** React Query client for cache invalidation */
  qc: QueryClient
  /** Current character data array for version lookup */
  data: any[]
}

export function useCharacterInlineEdit(deps: UseCharacterInlineEditDeps) {
  const { t, notification, qc, data } = deps

  const [inlineEdit, setInlineEdit] = React.useState<{
    id: string
    field: "name" | "description"
    value: string
    originalValue: string
  } | null>(null)
  const inlineSaveInFlightRef = React.useRef(false)
  const inlineEditInputRef = React.useRef<InputRef>(null)
  const inlineEditTriggerRef = React.useRef<HTMLElement | null>(null)
  const inlineEditFocusKeyRef = React.useRef<string | null>(null)

  const restoreInlineEditFocus = React.useCallback(() => {
    setTimeout(() => {
      const focusKey = inlineEditFocusKeyRef.current
      if (focusKey) {
        const target = document.querySelector<HTMLElement>(
          `[data-inline-edit-key="${focusKey}"]`
        )
        if (target) {
          target.focus()
          return
        }
      }
      inlineEditTriggerRef.current?.focus()
    }, 0)
  }, [])

  const { mutate: inlineUpdateCharacter, isPending: inlineUpdating } = useMutation({
    mutationFn: async ({
      id,
      field,
      value,
      version
    }: {
      id: string
      field: "name" | "description"
      value: string
      version?: number
    }) => {
      return await tldwClient.updateCharacter(id, { [field]: value }, version)
    },
    onSettled: () => {
      inlineSaveInFlightRef.current = false
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      setInlineEdit(null)
      restoreInlineEditFocus()
    },
    onError: (e: any) => {
      notification.error({
        message: t("settings:manageCharacters.notification.error", {
          defaultValue: "Error"
        }),
        description:
          e?.message ||
          t("settings:manageCharacters.notification.someError", {
            defaultValue: "Something went wrong"
          })
      })
    }
  })

  const startInlineEdit = React.useCallback(
    (record: any, field: "name" | "description", trigger?: HTMLElement | null) => {
      const id = String(record.id || record.slug || record.name)
      const value = record[field] || ""
      if (trigger) {
        inlineEditTriggerRef.current = trigger
      }
      inlineEditFocusKeyRef.current = `${id}:${field}`
      setInlineEdit({ id, field, value, originalValue: value })
      setTimeout(() => inlineEditInputRef.current?.focus(), 0)
    },
    []
  )

  const saveInlineEdit = React.useCallback(() => {
    if (!inlineEdit || inlineSaveInFlightRef.current) return
    const trimmedValue = inlineEdit.value.trim()

    // Validate name field
    if (inlineEdit.field === "name" && !trimmedValue) {
      notification.warning({
        message: t("settings:manageCharacters.form.name.required", {
          defaultValue: "Please enter a name"
        })
      })
      return
    }

    // Skip if unchanged
    if (trimmedValue === inlineEdit.originalValue) {
      setInlineEdit(null)
      return
    }

    // Find the record to get version
    const record = (data || []).find(
      (c: any) => String(c.id || c.slug || c.name) === inlineEdit.id
    )

    inlineSaveInFlightRef.current = true
    inlineUpdateCharacter({
      id: inlineEdit.id,
      field: inlineEdit.field,
      value: trimmedValue,
      version: record?.version
    })
  }, [inlineEdit, inlineUpdateCharacter, data, notification, t])

  const cancelInlineEdit = React.useCallback(() => {
    setInlineEdit(null)
    restoreInlineEditFocus()
  }, [restoreInlineEditFocus])

  return {
    // state
    inlineEdit,
    setInlineEdit,
    inlineUpdating,
    // refs
    inlineEditInputRef,
    inlineEditTriggerRef,
    inlineEditFocusKeyRef,
    // callbacks
    startInlineEdit,
    saveInlineEdit,
    cancelInlineEdit,
    restoreInlineEditFocus
  }
}
