import React from 'react'
import { Input } from 'antd'
import { Search as SearchIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { NOTE_SEARCH_DEBOUNCE_MS } from './notes-manager-utils'

type NotesTrashSearchProps = {
  /** The search the Trash list applies now. */
  value: string
  onSearch: (query: string) => void
}

/**
 * Search box for Trash (NL-05, #3101). It matches a title or content
 * substring on the server (`GET /api/v1/notes/trash?query=`), applied after
 * a short pause in typing or at once on Enter.
 */
export const NotesTrashSearch: React.FC<NotesTrashSearchProps> = ({ value, onSearch }) => {
  const { t } = useTranslation(['option'])
  const [input, setInput] = React.useState(value)
  const onSearchRef = React.useRef(onSearch)
  onSearchRef.current = onSearch

  // A search cleared from elsewhere (an empty state's Clear) clears the box.
  React.useEffect(() => {
    setInput(value)
  }, [value])

  React.useEffect(() => {
    if (input.trim() === value.trim()) return
    const timer = window.setTimeout(() => onSearchRef.current(input.trim()), NOTE_SEARCH_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [input, value])

  return (
    <Input
      allowClear
      size="small"
      placeholder={t('option:notesSearch.trashSearchPlaceholder', { defaultValue: 'Search Trash' })}
      aria-label={t('option:notesSearch.trashSearchAria', { defaultValue: 'Search Trash' })}
      prefix={<SearchIcon className="h-4 w-4 text-text-subtle" aria-hidden="true" />}
      value={input}
      data-testid="notes-trash-search"
      onChange={(event) => setInput(event.target.value)}
      onPressEnter={() => onSearch(input.trim())}
    />
  )
}

export default NotesTrashSearch
