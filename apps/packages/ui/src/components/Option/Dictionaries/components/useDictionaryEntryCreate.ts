import { useMutation, useQueryClient } from "@tanstack/react-query"
import React from "react"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  buildTimedEffectsPayload,
  validateRegexPattern,
} from "./dictionaryEntryUtils"

type UseDictionaryEntryCreateParams = {
  dictionaryId: number
  form: any
  allEntriesQueryKey: readonly [string, number]
  validateRegexWithServer: (entryDraft: any) => Promise<string | null>
}

type DictionaryEntryCreateState = {
  adding: boolean
  advancedMode: boolean
  toggleAdvancedMode: () => void
  regexError: string | null
  regexServerError: string | null
  handleAddEntrySubmit: (values: any) => Promise<void>
  handleAddEntryPatternChange: (patternValue: string) => void
  handleAddEntryReplacementChange: () => void
  handleAddEntryTypeChange: (value: string) => void
}

export function useDictionaryEntryCreate({
  dictionaryId,
  form,
  allEntriesQueryKey,
  validateRegexWithServer,
}: UseDictionaryEntryCreateParams): DictionaryEntryCreateState {
  const qc = useQueryClient()
  const notification = useAntdNotification()
  const [advancedMode, setAdvancedMode] = React.useState(false)
  const [regexError, setRegexError] = React.useState<string | null>(null)
  const [regexServerError, setRegexServerError] = React.useState<string | null>(null)

  const { mutate: addEntry, isPending: adding } = useMutation({
    mutationFn: (value: any) => tldwClient.addDictionaryEntry(dictionaryId, value),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ["tldw:listDictionaryEntries", dictionaryId] })
      qc.invalidateQueries({ queryKey: allEntriesQueryKey })
      const patternPreview = variables?.pattern || "Entry"
      notification.success({ message: `Entry "${patternPreview}" added` })
      form.resetFields()
      setRegexError(null)
      setRegexServerError(null)
    },
    onError: (error: any) => {
      const message = error?.message || "Failed to add entry."
      setRegexServerError(message)
      notification.error({ message: "Add entry failed", description: message })
    },
  })

  const handleAddEntrySubmit = React.useCallback(
    async (values: any) => {
      const entryType = form.getFieldValue("type") || "literal"
      setRegexServerError(null)
      if (entryType === "regex") {
        const pattern = form.getFieldValue("pattern")
        const error = validateRegexPattern(pattern)
        if (error) {
          setRegexError(error)
          return
        }

        const serverRegexError = await validateRegexWithServer({
          ...values,
          type: "regex",
        })
        if (serverRegexError) {
          setRegexServerError(serverRegexError)
          return
        }
      }

      const payload: Record<string, any> = { ...values }
      payload.case_sensitive =
        typeof values?.case_sensitive === "boolean" ? values.case_sensitive : false
      const timedEffectsPayload = buildTimedEffectsPayload(values?.timed_effects)
      if (timedEffectsPayload) {
        payload.timed_effects = timedEffectsPayload
      } else {
        delete payload.timed_effects
      }
      addEntry(payload)
    },
    [addEntry, form, validateRegexWithServer]
  )

  const handleAddEntryPatternChange = React.useCallback(
    (patternValue: string) => {
      setRegexServerError(null)
      const entryType = form.getFieldValue("type") || "literal"
      if (entryType === "regex") {
        const error = validateRegexPattern(patternValue)
        setRegexError(error)
      } else {
        setRegexError(null)
      }
    },
    [form]
  )

  const handleAddEntryReplacementChange = React.useCallback(() => {
    setRegexServerError(null)
  }, [])

  const handleAddEntryTypeChange = React.useCallback(
    (value: string) => {
      const pattern = form.getFieldValue("pattern")
      if (value === "regex" && pattern) {
        setRegexError(validateRegexPattern(pattern))
      } else {
        setRegexError(null)
      }
      setRegexServerError(null)
    },
    [form]
  )

  const toggleAdvancedMode = React.useCallback(() => {
    setAdvancedMode((current) => !current)
  }, [])

  return {
    adding,
    advancedMode,
    toggleAdvancedMode,
    regexError,
    regexServerError,
    handleAddEntrySubmit,
    handleAddEntryPatternChange,
    handleAddEntryReplacementChange,
    handleAddEntryTypeChange,
  }
}
