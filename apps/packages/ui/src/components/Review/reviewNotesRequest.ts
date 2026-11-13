import {
  buildNotesPageRequestPath,
  readNotesTotal,
  type NotesPageSort
} from '@/services/notes-page-request'
import type { MediaSortBy } from './mediaSearchRequest'

const NOTE_LIST_SORT: Record<MediaSortBy, NotesPageSort | null> = {
  relevance: null,
  date_desc: { sortBy: 'last_modified', sortOrder: 'desc' },
  date_asc: { sortBy: 'last_modified', sortOrder: 'asc' },
  title_asc: { sortBy: 'title', sortOrder: 'asc' },
  title_desc: { sortBy: 'title', sortOrder: 'desc' }
}

type ReviewNotesRequestArgs = {
  query: string
  keywordTokens: string[]
  page: number
  pageSize: number
  sortBy: MediaSortBy
}

/**
 * One page of Review's notes results (#3149). Review's tag filter keeps notes that
 * have every tag, as its media filter does; its sort applies to the plain list.
 */
export const buildReviewNotesRequestPath = ({
  query,
  keywordTokens,
  page,
  pageSize,
  sortBy
}: ReviewNotesRequestArgs) =>
  buildNotesPageRequestPath({ query, tokens: keywordTokens, page, pageSize, sort: NOTE_LIST_SORT[sortBy] })

export { readNotesTotal }
