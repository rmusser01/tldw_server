import React from "react"
import { Modal } from "antd"
import { useTranslation } from "react-i18next"
import { isFireFoxPrivateMode } from "@/utils/is-private-mode"
import { useAntdNotification } from "@/hooks/useAntdNotification"

export interface UseTemporaryChatToggleOptions {
  temporaryChat: boolean
  setTemporaryChat: (value: boolean) => void
  messagesLength: number
  clearChat: () => void
}

export interface UseTemporaryChatToggleResult {
  handleToggleTemporaryChat: (next: boolean) => void
  getPersistenceModeLabel: (isTemporary: boolean) => string
  getPersistenceModeChipLabel: (isTemporary: boolean) => string
  currentModeLabel: string
  currentModeChipLabel: string
}

/**
 * Hook that manages temporary chat toggle logic with confirmation modals.
 * Handles Firefox private mode detection, message clearing confirmations, etc.
 */
export const useTemporaryChatToggle = (
  options: UseTemporaryChatToggleOptions
): UseTemporaryChatToggleResult => {
  const { temporaryChat, setTemporaryChat, messagesLength, clearChat } = options
  const { t } = useTranslation(["playground", "sidepanel", "common"])
  const notification = useAntdNotification()

  const getPersistenceModeLabel = React.useCallback(
    (isTemporary: boolean) =>
      isTemporary
        ? t(
            "playground:composer.persistence.ephemeral",
            "Temporary chat: not saved in history and cleared when you close this window."
          )
        : t(
            "playground:composer.persistence.local",
            "Saved on this device only. This chat is not on your tldw server."
          ),
    [t]
  )

  const getPersistenceModeChipLabel = React.useCallback(
    (isTemporary: boolean) =>
      isTemporary
        ? t(
            "playground:composer.persistence.ephemeralShort",
            "Temporary (not saved)"
          )
        : t(
            "playground:composer.persistence.localShort",
            "Saved on this device"
          ),
    [t]
  )

  const handleToggleTemporaryChat = React.useCallback(
    (next: boolean) => {
      if (isFireFoxPrivateMode) {
        notification.error({
          message: t(
            "sidepanel:errors.privateModeTitle",
            "tldw Assistant can't save data"
          ),
          description: t(
            "sidepanel:errors.privateModeDescription",
            "Firefox Private Mode does not support saving chat history. Temporary chat is enabled by default. More fixes coming soon."
          )
        })
        return
      }

      const hadMessages = messagesLength > 0

      if (!next && temporaryChat && hadMessages) {
        notification.warning({
          message: t(
            "sidepanel:composer.privateChatLockedTitle",
            "Private chat is locked"
          ),
          description: t(
            "sidepanel:composer.privateChatLockedBody",
            "Start a new chat to switch back to saved conversations."
          )
        })
        return
      }

      // Show confirmation when enabling temporary mode with existing messages
      if (next && hadMessages) {
        Modal.confirm({
          title: t(
            "sidepanel:composer.tempChatConfirmTitle",
            "Enable temporary mode?"
          ),
          content: t(
            "sidepanel:composer.tempChatConfirmContent",
            "This will clear your current conversation. Messages won't be saved."
          ),
          okText: t("common:confirm", "Confirm"),
          cancelText: t("common:cancel", "Cancel"),
          onOk: () => {
            setTemporaryChat(next)
            clearChat()
            notification.info({
              message: t(
                "sidepanel:composer.tempChatClearedMessages",
                "Temporary chat enabled. Previous messages cleared."
              ),
              placement: "bottomRight",
              duration: 2.5
            })
          }
        })
        return
      }

      // Show confirmation when disabling temporary mode with existing messages
      if (!next && hadMessages) {
        Modal.confirm({
          title: t(
            "sidepanel:composer.tempChatDisableConfirmTitle",
            "Disable temporary mode?"
          ),
          content: t(
            "sidepanel:composer.tempChatDisableConfirmContent",
            "This will clear your current conversation. Messages will start saving again."
          ),
          okText: t("common:confirm", "Confirm"),
          cancelText: t("common:cancel", "Cancel"),
          onOk: () => {
            setTemporaryChat(next)
            clearChat()
            notification.info({
              message: t(
                "sidepanel:composer.tempChatDisabledClearedMessages",
                "Temporary chat disabled. Previous messages cleared."
              ),
              placement: "bottomRight",
              duration: 2.5
            })
          }
        })
        return
      }

      // No confirmation needed when toggling with no messages
      setTemporaryChat(next)
      if (hadMessages) {
        clearChat()
      }

      const modeLabel = getPersistenceModeLabel(next)
      notification.info({
        message: modeLabel,
        placement: "bottomRight",
        duration: 2.5
      })
    },
    [
      clearChat,
      getPersistenceModeLabel,
      messagesLength,
      notification,
      setTemporaryChat,
      temporaryChat,
      t
    ]
  )

  // Memoized labels for current state
  const currentModeLabel = React.useMemo(
    () => getPersistenceModeLabel(temporaryChat),
    [getPersistenceModeLabel, temporaryChat]
  )

  const currentModeChipLabel = React.useMemo(
    () => getPersistenceModeChipLabel(temporaryChat),
    [getPersistenceModeChipLabel, temporaryChat]
  )

  return {
    handleToggleTemporaryChat,
    getPersistenceModeLabel,
    getPersistenceModeChipLabel,
    currentModeLabel,
    currentModeChipLabel
  }
}
