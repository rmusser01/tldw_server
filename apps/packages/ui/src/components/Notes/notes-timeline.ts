import type { NoteListItem } from './notes-manager-types'
import { toSortableTimestamp } from './notes-manager-utils'

export type NotesTimelineSection = {
  key: string
  label: string
  notes: NoteListItem[]
}

export const NOTES_TIMELINE_UNKNOWN_KEY = 'unknown'

/** `YYYY-MM` for the note's month in the reader's local time zone. */
export const toLocalTimelineMonthKey = (timestamp: number): string => {
  const date = new Date(timestamp)
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
}

/** "October 2026" for a `YYYY-MM` key, built in local time so it matches the key. */
export const formatTimelineMonthLabel = (key: string, locale?: string | string[]): string => {
  const match = /^(\d{4})-(\d{2})$/.exec(key)
  if (!match) return key
  const date = new Date(Number(match[1]), Number(match[2]) - 1, 1)
  if (Number.isNaN(date.getTime())) return key
  return date.toLocaleDateString(locale, { month: 'long', year: 'numeric' })
}

/**
 * Group notes into month sections, newest month first (NL-09).
 *
 * Both the bucket key and its label use local time. Bucketing by UTC month and
 * labelling in local time filed early-month notes under the previous month for
 * readers west of UTC.
 */
export const buildNotesTimelineSections = (
  notes: readonly NoteListItem[],
  options: { unknownLabel: string; locale?: string | string[] }
): NotesTimelineSection[] => {
  const bucketMap = new Map<string, NoteListItem[]>()
  for (const note of notes) {
    const timestamp = toSortableTimestamp(note.updated_at)
    const key = timestamp > 0 ? toLocalTimelineMonthKey(timestamp) : NOTES_TIMELINE_UNKNOWN_KEY
    const bucket = bucketMap.get(key) || []
    bucket.push(note)
    bucketMap.set(key, bucket)
  }
  const orderedKeys = Array.from(bucketMap.keys()).sort((a, b) => {
    if (a === NOTES_TIMELINE_UNKNOWN_KEY) return 1
    if (b === NOTES_TIMELINE_UNKNOWN_KEY) return -1
    return b.localeCompare(a)
  })
  return orderedKeys.map((key) => {
    const notesForKey = bucketMap.get(key) || []
    notesForKey.sort(
      (a, b) => toSortableTimestamp(b.updated_at) - toSortableTimestamp(a.updated_at)
    )
    const label =
      key === NOTES_TIMELINE_UNKNOWN_KEY
        ? options.unknownLabel
        : formatTimelineMonthLabel(key, options.locale)
    return { key, label, notes: notesForKey }
  })
}
