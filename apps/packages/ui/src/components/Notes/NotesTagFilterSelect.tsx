import React from 'react'
import { Select } from 'antd'
import { useTranslation } from 'react-i18next'

type NotesTagFilterSelectProps = {
  value: string[]
  options: string[]
  renderOptionLabel: (keyword: string) => React.ReactNode
  onSearch: (text: string) => void
  onChange: (values: string[]) => void
}

/** Case-insensitive "contains" match on the tag itself, not its rendered label. */
export const matchesTagFilterQuery = (input: string, tag: unknown): boolean => {
  const query = input.trim().toLowerCase()
  if (!query) return true
  return String(tag ?? '').toLowerCase().includes(query)
}

/**
 * "Filter by tag" picker for the notes list (NL-08).
 *
 * Only existing tags can be chosen: typing narrows the list and Enter picks
 * the first match. Free text used to become a filter of its own, so a partial
 * word like "thes" filtered the list down to nothing and then showed up in
 * Browse tags as if it were a real tag.
 */
const NotesTagFilterSelect: React.FC<NotesTagFilterSelectProps> = ({
  value,
  options,
  renderOptionLabel,
  onSearch,
  onChange
}) => {
  const { t } = useTranslation(['option'])
  const [searchText, setSearchText] = React.useState('')
  const trimmedSearch = searchText.trim()

  return (
    <Select
      mode="multiple"
      showSearch
      allowClear
      placeholder={t('option:notesSearch.keywordsPlaceholder', {
        defaultValue: 'Filter by tag'
      })}
      aria-label={t('option:notesSearch.keywordFilterAriaLabel', {
        defaultValue: 'Filter by tag'
      })}
      className="w-full"
      value={value}
      searchValue={searchText}
      filterOption={(input, option) => matchesTagFilterQuery(input, option?.value)}
      onSearch={(text) => {
        setSearchText(text)
        onSearch(text)
      }}
      onChange={(next) => {
        setSearchText('')
        onChange(Array.isArray(next) ? next : [next])
      }}
      onBlur={() => setSearchText('')}
      notFoundContent={
        <span className="text-xs text-text-muted" data-testid="notes-tag-filter-no-match">
          {trimmedSearch
            ? t('option:notesSearch.tagFilterNoMatch', {
                defaultValue: 'No tag named "{{query}}"',
                query: trimmedSearch
              }).replace('{{query}}', () => trimmedSearch)
            : t('option:notesSearch.tagFilterNoTags', {
                defaultValue: 'No tags yet'
              })}
        </span>
      }
      options={options.map((keyword) => ({
        label: renderOptionLabel(keyword),
        value: keyword
      }))}
    />
  )
}

export default NotesTagFilterSelect
