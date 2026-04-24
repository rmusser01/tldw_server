import React from "react"
import { notification } from "antd"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { isFireFoxPrivateMode } from "@/utils/is-private-mode"

export interface UsePromptUtilitiesDeps {
  t: (key: string, opts?: Record<string, any>) => string
  data: any[] | undefined
}

export function usePromptUtilities(deps: UsePromptUtilitiesDeps) {
  const { t, data } = deps

  const confirmDanger = useConfirmDanger()

  const guardPrivateMode = React.useCallback(() => {
    if (!isFireFoxPrivateMode) return false
    notification.error({
      message: t(
        "common:privateModeSaveErrorTitle",
        "tldw Assistant can't save data"
      ),
      description: t(
        "settings:prompts.privateModeDescription",
        "Firefox Private Mode does not support saving data to IndexedDB. Please add prompts from a normal window."
      )
    })
    return true
  }, [isFireFoxPrivateMode, t])

  const getPromptKeywords = React.useCallback(
    (prompt: any) => prompt?.keywords ?? prompt?.tags ?? [],
    []
  )

  const getPromptTexts = React.useCallback((prompt: any) => {
    const systemText =
      prompt?.system_prompt ||
      (prompt?.is_system ? prompt?.content : undefined)
    const userText =
      prompt?.user_prompt ||
      (!prompt?.is_system ? prompt?.content : undefined)
    return { systemText, userText }
  }, [])

  const getPromptType = React.useCallback((prompt: any) => {
    const { systemText, userText } = getPromptTexts(prompt)
    const hasSystem = typeof systemText === "string" && systemText.trim().length > 0
    const hasUser = typeof userText === "string" && userText.trim().length > 0
    if (hasSystem && hasUser) return "mixed"
    if (hasSystem) return "system"
    if (hasUser) return "quick"
    return prompt?.is_system ? "system" : "quick"
  }, []) // getPromptTexts has stable identity (empty deps), safe to omit

  const getPromptModifiedAt = React.useCallback((prompt: any) => {
    return prompt?.updatedAt || prompt?.createdAt || 0
  }, [])

  const getPromptUsageCount = React.useCallback((prompt: any) => {
    const value = prompt?.usageCount
    if (typeof value !== "number" || Number.isNaN(value)) return 0
    return Math.max(0, Math.floor(value))
  }, [])

  const getPromptLastUsedAt = React.useCallback((prompt: any) => {
    const value = prompt?.lastUsedAt
    if (typeof value !== "number" || Number.isNaN(value)) return null
    return value
  }, [])

  const formatRelativePromptTime = React.useCallback(
    (timestamp: number | null | undefined) => {
      if (!timestamp) {
        return t("common:unknown", { defaultValue: "Unknown" })
      }
      const now = Date.now()
      const diffMs = Math.max(0, now - timestamp)
      const diffMins = Math.floor(diffMs / (1000 * 60))
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

      if (diffMins < 1) {
        return t("common:justNow", { defaultValue: "Just now" })
      }
      if (diffMins < 60) {
        return t("common:minutesAgo", {
          defaultValue: "{{count}}m ago",
          count: diffMins
        })
      }
      if (diffHours < 24) {
        return t("common:hoursAgo", {
          defaultValue: "{{count}}h ago",
          count: diffHours
        })
      }
      if (diffDays < 30) {
        return t("common:daysAgo", {
          defaultValue: "{{count}}d ago",
          count: diffDays
        })
      }
      return new Date(timestamp).toLocaleDateString()
    },
    [t]
  )

  const getPromptRecordById = React.useCallback(
    (promptId: string) => {
      const prompts = Array.isArray(data) ? data : []
      return prompts.find((prompt: any) => String(prompt?.id) === String(promptId))
    },
    [data]
  )

  return React.useMemo(
    () => ({
      confirmDanger,
      guardPrivateMode,
      getPromptKeywords,
      getPromptTexts,
      getPromptType,
      getPromptModifiedAt,
      getPromptUsageCount,
      getPromptLastUsedAt,
      formatRelativePromptTime,
      getPromptRecordById,
      isFireFoxPrivateMode
    }),
    [
      confirmDanger,
      guardPrivateMode,
      getPromptKeywords,
      getPromptTexts,
      getPromptType,
      getPromptModifiedAt,
      getPromptUsageCount,
      getPromptLastUsedAt,
      formatRelativePromptTime,
      getPromptRecordById
    ]
  )
}
