import React from 'react'
import { Button } from 'antd'

type ToastApi = {
  success: (content: React.ReactNode) => unknown
  open?: (config: { key: string; type: 'success'; duration: number; content: React.ReactNode }) => unknown
  destroy?: (key?: string) => unknown
}

const NOTE_RESTORED_TOAST_KEY = 'notes-restored'

type NoteRestoredToastOptions = {
  noteId: string
  label: string
  openLabel: string
  onOpen: () => void
}

/**
 * "Note restored · Open note" (NL-05, #3101). Restoring from Trash keeps the
 * user in Trash, so they can restore several notes in a row; the toast's
 * Open note takes them to the restored note when they want it.
 */
export const showNoteRestoredToast = (message: ToastApi, { noteId, label, openLabel, onOpen }: NoteRestoredToastOptions) => {
  if (typeof message.open !== 'function') {
    message.success(label)
    return
  }
  // One key: restoring several notes in a row keeps one toast, for the latest note.
  const key = NOTE_RESTORED_TOAST_KEY
  message.open({
    key,
    type: 'success',
    duration: 8,
    content: (
      <span className="inline-flex items-center gap-2">
        <span>{label}</span>
        <Button
          type="link"
          size="small"
          className="!px-0"
          data-testid="notes-restored-open-note"
          onClick={() => {
            message.destroy?.(key)
            onOpen()
          }}
        >
          {openLabel}
        </Button>
      </span>
    )
  })
}
