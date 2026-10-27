import React from "react"
import type { TFunction } from "i18next"
import { message } from "antd"

type UseBulkChatOperationsParams = {
  selectedConversationIds: string[]
  folderApiAvailable: boolean | null
  deleteConversation: (conversationId: string) => Promise<void>
  t: TFunction
  setBulkFolderPickerOpen: (open: boolean) => void
  setBulkTagPickerOpen: (open: boolean) => void
}

export type BulkDeleteResult = {
  deletedConversationIds: Set<string>
  failedConversationIds: Set<string>
}

type RunBulkDeleteParams = {
  selectedConversationIds: string[]
  deleteConversation: (conversationId: string) => Promise<void>
  t: TFunction
  isCurrent?: () => boolean
}

export const runBulkDelete = async ({
  selectedConversationIds,
  deleteConversation,
  t,
  isCurrent = () => true
}: RunBulkDeleteParams): Promise<BulkDeleteResult | null> => {
  if (selectedConversationIds.length === 0 || !isCurrent()) return null

  const results = await Promise.allSettled(
    selectedConversationIds.map((conversationId) =>
      isCurrent()
        ? deleteConversation(conversationId)
        : Promise.reject(new Error("server_history_action_expired"))
    )
  )
  if (!isCurrent()) return null
  const deletedConversationIds = new Set<string>()
  const failedConversationIds = new Set<string>()
  let failures = 0
  results.forEach((result, index) => {
    const conversationId = selectedConversationIds[index]
    if (result.status === "rejected") {
      failures += 1
      failedConversationIds.add(conversationId)
    } else {
      deletedConversationIds.add(conversationId)
    }
  })

  if (failures === 0) {
    message.success(
      t("sidepanel:multiSelect.deleteSuccess", "Chats moved to trash.")
    )
  } else if (failures === selectedConversationIds.length) {
    message.error(
      t(
        "sidepanel:multiSelect.deleteFailed",
        "Unable to move selected chats to trash."
      )
    )
  } else {
    message.error(
      t(
        "sidepanel:multiSelect.deletePartial",
        "Some chats could not be moved to trash."
      )
    )
  }

  return { deletedConversationIds, failedConversationIds }
}

export const useBulkChatOperations = ({
  selectedConversationIds,
  folderApiAvailable,
  deleteConversation,
  t,
  setBulkFolderPickerOpen,
  setBulkTagPickerOpen
}: UseBulkChatOperationsParams) => {
  const openBulkFolderPicker = React.useCallback(() => {
    if (folderApiAvailable === false) {
      message.error(
        t(
          "sidepanel:folderPicker.notAvailable",
          "Folder organization is not available on this server"
        )
      )
      return
    }
    if (selectedConversationIds.length === 0) {
      message.warning(
        t(
          "sidepanel:multiSelect.serverOnlyWarning",
          "Select chats saved on the server to apply this action."
        )
      )
      return
    }
    setBulkFolderPickerOpen(true)
  }, [folderApiAvailable, selectedConversationIds, setBulkFolderPickerOpen, t])

  const openBulkTagPicker = React.useCallback(() => {
    if (folderApiAvailable === false) {
      message.error(
        t(
          "sidepanel:multiSelect.tagsUnavailable",
          "Tags are not available on this server"
        )
      )
      return
    }
    if (selectedConversationIds.length === 0) {
      message.warning(
        t(
          "sidepanel:multiSelect.serverOnlyWarning",
          "Select chats saved on the server to apply this action."
        )
      )
      return
    }
    setBulkTagPickerOpen(true)
  }, [folderApiAvailable, selectedConversationIds, setBulkTagPickerOpen, t])

  const applyBulkDelete = React.useCallback(
    async (isCurrent?: () => boolean): Promise<BulkDeleteResult | null> => {
      return runBulkDelete({
        selectedConversationIds,
        deleteConversation,
        t,
        isCurrent
      })
    },
    [deleteConversation, selectedConversationIds, t]
  )

  return {
    openBulkFolderPicker,
    openBulkTagPicker,
    applyBulkDelete
  }
}
