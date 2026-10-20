import React from 'react'
import { Button, Select, Spin, Tooltip, Typography } from 'antd'
import { useTranslation } from 'react-i18next'
import type { MoodboardSummary } from './notes-manager-types'

export interface NotesCollectionControlsProps {
  moodboards: MoodboardSummary[]
  selectedMoodboardId: number | null
  selectedMoodboard: MoodboardSummary | null
  isMoodboardsFetching: boolean
  /** The note open in the editor, or null when nothing saved is open. */
  openNoteId: string | number | null
  /** True when the open note already shows in this collection. */
  openNoteInCollection: boolean
  setSelectedMoodboardId: (id: number | null) => void
  setPage: React.Dispatch<React.SetStateAction<number>>
  createMoodboard: () => Promise<void>
  renameMoodboard: () => Promise<void>
  deleteMoodboard: () => Promise<void>
  addNoteToMoodboard: (noteId: string | number) => Promise<void>
}

/**
 * Collection picker and actions shown at the top of the Collection view
 * (NL-13). They used to sit in the collapsed Organize section, so the
 * Collection view said "Create or select a collection" with no way to do
 * either. "Add open note" is the only way to put a note in a collection
 * from the Notes page.
 */
const NotesCollectionControls: React.FC<NotesCollectionControlsProps> = ({
  moodboards,
  selectedMoodboardId,
  selectedMoodboard,
  isMoodboardsFetching,
  openNoteId,
  openNoteInCollection,
  setSelectedMoodboardId,
  setPage,
  createMoodboard,
  renameMoodboard,
  deleteMoodboard,
  addNoteToMoodboard
}) => {
  const { t } = useTranslation(['option'])
  const addDisabledReason =
    openNoteId == null
      ? t('option:notesSearch.moodboardAddOpenNoteNeedsNote', {
          defaultValue: 'Open a note first'
        })
      : openNoteInCollection
        ? t('option:notesSearch.moodboardAddOpenNoteAlreadyIn', {
            defaultValue: 'The open note is already in this collection'
          })
        : null

  return (
    <div
      className="mb-3 space-y-2 rounded border border-border bg-surface2 p-2"
      data-testid="notes-moodboard-controls"
    >
      <div className="flex items-center justify-between">
        <Typography.Text className="text-[11px] uppercase tracking-[0.08em] text-text-muted">
          {t('option:notesSearch.moodboardLabel', {
            defaultValue: 'Collection'
          })}
        </Typography.Text>
        {isMoodboardsFetching && <Spin size="small" />}
      </div>
      <Select
        className="w-full"
        size="small"
        aria-label={t('option:notesSearch.moodboardSelectAriaLabel', {
          defaultValue: 'Collection'
        })}
        value={selectedMoodboardId == null ? undefined : selectedMoodboardId}
        onChange={(value) => {
          if (value == null) {
            setSelectedMoodboardId(null)
            setPage(1)
            return
          }
          const parsed = Number(value)
          if (!Number.isFinite(parsed)) return
          setSelectedMoodboardId(Math.floor(parsed))
          setPage(1)
        }}
        placeholder={t('option:notesSearch.moodboardEmptyOption', {
          defaultValue: 'No collections yet'
        })}
        options={moodboards.map((board) => ({
          value: board.id,
          label: board.name
        }))}
        data-testid="notes-moodboard-select"
      />
      <div className="grid grid-cols-3 gap-2">
        <Button
          size="small"
          onClick={() => {
            void createMoodboard()
          }}
          aria-label={t('option:notesSearch.moodboardCreateAction', {
            defaultValue: 'New collection'
          })}
          data-testid="notes-moodboard-create"
        >
          {t('option:notesSearch.moodboardCreate', {
            defaultValue: 'New'
          })}
        </Button>
        <Button
          size="small"
          onClick={() => {
            void renameMoodboard()
          }}
          disabled={selectedMoodboard == null}
          data-testid="notes-moodboard-rename"
        >
          {t('option:notesSearch.moodboardRename', {
            defaultValue: 'Rename'
          })}
        </Button>
        <Button
          size="small"
          danger
          onClick={() => {
            void deleteMoodboard()
          }}
          disabled={selectedMoodboard == null}
          data-testid="notes-moodboard-delete"
        >
          {t('option:notesSearch.moodboardDelete', {
            defaultValue: 'Delete'
          })}
        </Button>
      </div>
      {selectedMoodboard != null && (
        <Tooltip title={addDisabledReason ?? undefined}>
          <span className="block">
            <Button
              size="small"
              block
              onClick={() => {
                if (openNoteId == null) return
                void addNoteToMoodboard(openNoteId)
              }}
              disabled={addDisabledReason != null}
              data-testid="notes-moodboard-add-open-note"
            >
              {t('option:notesSearch.moodboardAddOpenNote', {
                defaultValue: 'Add open note'
              })}
            </Button>
          </span>
        </Tooltip>
      )}
      {selectedMoodboard?.description ? (
        <Typography.Text type="secondary" className="block text-[11px] text-text-muted">
          {String(selectedMoodboard.description)}
        </Typography.Text>
      ) : null}
    </div>
  )
}

export default NotesCollectionControls
