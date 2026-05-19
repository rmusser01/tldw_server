import { useMutation } from "@tanstack/react-query"
import React from "react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  getActiveFocusableElement,
  restoreFocusToElement,
} from "./focusUtils"
import {
  buildTimedEffectsPayload,
  validateRegexPattern,
} from "./dictionaryEntryUtils"

type UseDictionaryEntryEditParams = {
  dictionaryId: number
  allEntriesQueryKey: readonly unknown[]
  editEntryForm: any
  notification: {
    success: (config: { message: string; description?: string }) => void
    error: (config: { message: string; description?: string }) => void
  }
  queryClient: {
    invalidateQueries: (input: { queryKey: readonly unknown[] }) => Promise<unknown>
  }
  validateRegexWithServer: (entryDraft: any) => Promise<string | null>
}

type UseDictionaryEntryEditResult = {
  editingEntry: any | null
  updatingEntry: boolean
  openEditEntryPanel: (entry: any) => void
  closeEditEntryPanel: () => void
  handleEditEntrySubmit: (values: any) => Promise<void>
}

export function useDictionaryEntryEdit({
  dictionaryId,
  allEntriesQueryKey,
  editEntryForm,
  notification,
  queryClient,
  validateRegexWithServer,
}: UseDictionaryEntryEditParams): UseDictionaryEntryEditResult {
  const [editingEntry, setEditingEntry] = React.useState<any | null>(null)
  const editEntryFocusReturnRef = React.useRef<HTMLElement | null>(null)
  const { t } = useTranslation(["option"])
  const localize = React.useCallback(
    (key: string, fallback: string) => t(key, fallback),
    [t]
  )

  React.useEffect(() => {
    if (editingEntry) return
    const focusTarget = editEntryFocusReturnRef.current
    editEntryFocusReturnRef.current = null
    restoreFocusToElement(focusTarget)
  }, [editingEntry])

  const { mutateAsync: updateEntry, isPending: updatingEntry } = useMutation({
    mutationFn: ({ entryId, data }: { entryId: number; data: any }) =>
      tldwClient.updateDictionaryEntry(entryId, data),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["tldw:listDictionaryEntries", dictionaryId],
      })
      await queryClient.invalidateQueries({ queryKey: allEntriesQueryKey })
      setEditingEntry(null)
      editEntryForm.resetFields()
      notification.success({
        message: t("option:dictionaries.entryUpdated", "Entry updated")
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("option:dictionaries.updateEntryFailedTitle", "Update failed"),
        description:
          error?.message ||
          t(
            "option:dictionaries.updateEntryFailedDescription",
            "Unable to update entry."
          ),
      })
    },
  })

  const closeEditEntryPanel = React.useCallback(() => {
    setEditingEntry(null)
    editEntryForm.resetFields()
  }, [editEntryForm])

  const openEditEntryPanel = React.useCallback(
    (entry: any) => {
      editEntryFocusReturnRef.current = getActiveFocusableElement()
      setEditingEntry(entry)
      editEntryForm.setFieldsValue({
        ...entry,
        timed_effects: buildTimedEffectsPayload(entry?.timed_effects, {
          forceObject: true,
        }),
      })
      editEntryForm.setFields([{ name: "pattern", errors: [] }])
    },
    [editEntryForm]
  )

  const handleEditEntrySubmit = React.useCallback(
    async (values: any) => {
      if (!editingEntry?.id) return

      const entryType = values?.type === "regex" ? "regex" : "literal"
      const pattern = typeof values?.pattern === "string" ? values.pattern : ""
      if (entryType === "regex") {
        const regexValidationError = validateRegexPattern(pattern, localize)
        if (regexValidationError) {
          editEntryForm.setFields([{ name: "pattern", errors: [regexValidationError] }])
          return
        }

        const serverRegexError = await validateRegexWithServer(values)
        if (serverRegexError) {
          editEntryForm.setFields([{ name: "pattern", errors: [serverRegexError] }])
          return
        }
      }

      const payload: Record<string, any> = {
        ...values,
        timed_effects: buildTimedEffectsPayload(values?.timed_effects, {
          forceObject: true,
        }),
      }

      try {
        await updateEntry({ entryId: editingEntry.id, data: payload })
      } catch (error: any) {
        const message =
          error?.message ||
          t(
            "option:dictionaries.updateEntryFailedTitle",
            "Update failed"
          )
        if (/regex|pattern|dangerous/i.test(message)) {
          editEntryForm.setFields([{ name: "pattern", errors: [message] }])
        }
      }
    },
    [editEntryForm, editingEntry?.id, localize, t, updateEntry, validateRegexWithServer]
  )

  return {
    editingEntry,
    updatingEntry,
    openEditEntryPanel,
    closeEditEntryPanel,
    handleEditEntrySubmit,
  }
}
