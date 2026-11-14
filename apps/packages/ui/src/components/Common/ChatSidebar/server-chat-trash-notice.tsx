import React from "react"
import type { TFunction } from "i18next"
import { message } from "antd"

let noticeSequence = 0

/**
 * "Moved to Trash" with Undo and View Trash (CS-10). Moving a chat to Trash
 * used to make its row vanish without a word.
 */
export const showMovedToTrashNotice = ({
  t,
  onUndo,
  onViewTrash
}: {
  t: TFunction
  onUndo: () => void
  onViewTrash: () => void
}) => {
  noticeSequence += 1
  const key = `server-chat-moved-to-trash-${noticeSequence}`
  const close = () => message.destroy?.(key)
  const actionClassName =
    "rounded px-1 font-medium text-primary hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
  message.success({
    key,
    duration: 8,
    content: (
      <span className="inline-flex flex-wrap items-center gap-2">
        <span>
          {t("common:chatSidebar.movedToTrash", { defaultValue: "Moved to Trash" })}
        </span>
        <button
          type="button"
          className={actionClassName}
          onClick={() => {
            close()
            onUndo()
          }}>
          {t("common:chatSidebar.undoTrash", { defaultValue: "Undo" })}
        </button>
        <button
          type="button"
          className={actionClassName}
          onClick={() => {
            close()
            onViewTrash()
          }}>
          {t("common:chatSidebar.viewTrash", { defaultValue: "View Trash" })}
        </button>
      </span>
    )
  })
}
