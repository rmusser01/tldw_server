import {
  applyChatSettingsPatch,
  chatSettingsStorageForKey,
  getChatSettingsForKey,
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey
} from "@/services/chat-settings"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"
import React from "react"

import { useStorage } from "@plasmohq/storage/hook"

import { useHistorySelectionContext } from "./useHistorySelection"

type UseChatSettingsRecordParams = {
  historyId: string | null
  serverChatId: string | null
}

export const useChatSettingsRecord = ({
  historyId,
  serverChatId
}: UseChatSettingsRecordParams) => {
  const controller = useHistorySelectionContext()
  const mode = controller?.settingsMode?.(serverChatId) ?? "ordinary"
  const guarded = mode !== "ordinary"
  const stableHistoryId = historyId && historyId !== "temp" ? historyId : null
  const chatKey = React.useMemo(
    () =>
      guarded
        ? "h1-fork-settings-presentation"
        : resolveChatSettingsKey({ historyId: stableHistoryId, serverChatId }),
    [serverChatId, stableHistoryId, guarded]
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
    if (!guarded) void getChatSettingsForKey(chatKey)
  }, [chatKey, guarded])
  const settings = React.useMemo(
    () =>
      guarded
        ? mode === "fork"
          ? (controller?.forkSettings ?? null)
          : null
        : normalizeChatSettingsRecord(rawSettings),
    [rawSettings, guarded, mode, controller?.forkSettings]
  )

  const updateSettings = React.useCallback(
    async (patch: Partial<ChatSettingsRecord>) => {
      if (mode === "pending") throw new Error("fork_settings_owner_unavailable")
      if (mode === "fork") return controller!.updateForkSettings(patch)
      const next = await applyChatSettingsPatch({
        historyId: stableHistoryId,
        serverChatId,
        patch
      })
      return next
    },
    [serverChatId, stableHistoryId, mode, controller]
  )

  return { settings, updateSettings, chatKey }
}
