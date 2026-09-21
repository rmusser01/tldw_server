import React from "react"
import { Button } from "antd"
import { Undo2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"

export interface UndoNotificationOptions {
  /** Title of the notification */
  title: string
  /** Description of what was deleted */
  description?: string
  /** Duration in seconds before auto-dismiss (default: 8) */
  duration?: number
  /** Callback when undo is clicked */
  onUndo: () => Promise<void> | void
  /** Callback when notification is dismissed without undo */
  onDismiss?: () => void
}

/**
 * Hook for showing undo notifications after destructive actions.
 * Provides a way to undo deletions within a time window.
 *
 * @example
 * const { showUndoNotification } = useUndoNotification()
 *
 * // When deleting a chat
 * const deletedChat = chatToDelete
 * deleteChat(id)
 * showUndoNotification({
 *   title: "Chat deleted",
 *   description: "The conversation has been removed",
 *   onUndo: async () => {
 *     await restoreChat(deletedChat)
 *   }
 * })
 */
export function useUndoNotification() {
  const { t } = useTranslation(["common"])
  const notification = useAntdNotification()

  const showUndoNotification = ({
    title,
    description,
    duration = 8,
    onUndo,
    onDismiss
  }: UndoNotificationOptions) => {
    const key = `undo-${Date.now()}`
    let undoClicked = false

    const handleUndo = async () => {
      if (undoClicked) return
      undoClicked = true
      notification.destroy(key)
      try {
        await onUndo()
        notification.success({
          message: t("common:undo.restored", "Restored successfully"),
          duration: 2,
          placement: "bottomRight"
        })
      } catch (err) {
        notification.error({
          message: t("common:undo.restoreFailed", "Failed to restore"),
          description: err instanceof Error ? err.message : undefined,
          duration: 4,
          placement: "bottomRight"
        })
      }
    }

    notification.open({
      key,
      message: (
        <span className="font-medium text-text">
          {title}
        </span>
      ),
      description: description ? (
        <span className="text-text-muted">{description}</span>
      ) : undefined,
      icon: null,
      duration,
      placement: "bottomRight",
      className: "undo-notification",
      actions: (
        <Button
          type="primary"
          size="small"
          icon={<Undo2 className="size-3" />}
          onClick={handleUndo}
          className="flex items-center gap-1"
        >
          {t("common:undo.action", "Undo")}
        </Button>
      ),
      onClose: () => {
        if (!undoClicked && onDismiss) {
          onDismiss()
        }
      }
    })

    return key
  }

  return {
    showUndoNotification
  }
}

export default useUndoNotification
