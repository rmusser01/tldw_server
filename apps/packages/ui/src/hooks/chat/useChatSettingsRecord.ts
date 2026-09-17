import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import {
  applyChatSettingsPatch,
  chatSettingsStorageForKey,
  getChatSettingsForKey,
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey
} from "@/services/chat-settings"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"

type UseChatSettingsRecordParams = {
  historyId: string | null
  serverChatId: string | null
}

export const useChatSettingsRecord = ({
  historyId,
  serverChatId
}: UseChatSettingsRecordParams) => {
  const stableHistoryId = historyId && historyId !== "temp" ? historyId : null
  const chatKey = React.useMemo(
    () => resolveChatSettingsKey({ historyId: stableHistoryId, serverChatId }),
    [serverChatId, stableHistoryId]
  )
  const storageKey = React.useMemo(
    () => getChatSettingsStorageKey(chatKey),
    [chatKey]
  )

  const [rawSettings] = useStorage<ChatSettingsRecord | null | undefined>({
    key: storageKey,
    instance: chatSettingsStorageForKey(chatKey)
  })
  React.useEffect(() => {
    void getChatSettingsForKey(chatKey)
  }, [chatKey])
  const settings = React.useMemo(
    () => normalizeChatSettingsRecord(rawSettings),
    [rawSettings]
  )

  const updateSettings = React.useCallback(
    async (patch: Partial<ChatSettingsRecord>) => {
      const next = await applyChatSettingsPatch({
        historyId: stableHistoryId,
        serverChatId,
        patch
      })
      return next
    },
    [serverChatId, stableHistoryId]
  )

  return { settings, updateSettings, chatKey }
}
