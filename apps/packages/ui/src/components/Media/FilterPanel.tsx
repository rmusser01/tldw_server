import { ChevronDown, Filter, Plus, Star, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Select } from 'antd'
import { useTranslation } from 'react-i18next'
import type {
  MediaBoostFields,
  MediaDateRange,
  MediaSearchField,
  MediaSearchMode,
  MediaSortBy
} from '@/components/Review/mediaSearchRequest'
import {
  DEFAULT_MEDIA_SEARCH_FIELDS,
  hasDefaultMediaSearchFields,
  normalizeMediaSearchFields
} from '@/components/Review/mediaSearchRequest'
import type {
  MetadataMatchMode,
  MetadataSearchFilter,
  MetadataSearchField,
  MetadataSearchOperator
} from '@/components/Review/mediaMetadataSearchRequest'
import {
  createMetadataSearchFilter,
  getAllowedMetadataOperators,
  METADATA_SEARCH_FIELDS,
  METADATA_SEARCH_OPERATORS,
  normalizeMetadataSearchFilters
} from '@/components/Review/mediaMetadataSearchRequest'

interface FilterPanelProps {
  searchMode?: MediaSearchMode
  onSearchModeChange?: (mode: MediaSearchMode) => void
  mediaTypes: string[]
  selectedMediaTypes: string[]
  onMediaTypesChange: (types: string[]) => void
  sortBy?: MediaSortBy
  onSortByChange?: (sortBy: MediaSortBy) => void
  dateRange?: MediaDateRange
  onDateRangeChange?: (range: MediaDateRange) => void
  exactPhrase?: string
  onExactPhraseChange?: (value: string) => void
  searchFields?: MediaSearchField[]
  onSearchFieldsChange?: (fields: MediaSearchField[]) => void
  enableBoostFields?: boolean
  onEnableBoostFieldsChange?: (enabled: boolean) => void
  boostFields?: MediaBoostFields
  onBoostFieldsChange?: (fields: MediaBoostFields) => void
  metadataFilters?: MetadataSearchFilter[]
  onMetadataFiltersChange?: (filters: MetadataSearchFilter[]) => void
  metadataMatchMode?: MetadataMatchMode
  onMetadataMatchModeChange?: (mode: MetadataMatchMode) => void
  metadataValidationError?: string | null
  selectedKeywords: string[]
  onKeywordsChange: (keywords: string[]) => void
  selectedExcludedKeywords?: string[]
  onExcludedKeywordsChange?: (keywords: string[]) => void
  keywordOptions?: string[]
  keywordSourceMode?: 'endpoint' | 'results'
  onKeywordSearch?: (text: string) => void
  showFavoritesOnly?: boolean
  onShowFavoritesOnlyChange?: (show: boolean) => void
  favoritesCount?: number
  activeFilterCount?: number
  onClearAll?: () => void
}

// Normalize media type to Title Case for consistent display
const toTitleCase = (str: string): string => {
  if (!str) return str
  return str
    .toLowerCase()
    .split(/[\s_-]+/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

// Get user-friendly label for media type
const getMediaTypeLabel = (type: string): string => {
  const normalized = toTitleCase(type)
  // Map common types to better labels if needed
  const labelMap: Record<string, string> = {
    Youtube: 'YouTube',
    Pdf: 'PDF',
    Mp3: 'MP3',
    Mp4: 'MP4',
    Wav: 'WAV',
    Html: 'HTML',
    Url: 'URL'
  }
  return labelMap[normalized] || normalized
}

const DATE_INPUT_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/

const formatDateInputValue = (value?: string | null): string => {
  if (!value) return ''

  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return ''

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

const parseDateInputValue = (
  value: string,
  boundary: 'start' | 'end'
): string | null => {
  if (!value) return null

  const match = DATE_INPUT_PATTERN.exec(value)
  if (!match) return null

  const [, yearText, monthText, dayText] = match
  const year = Number(yearText)
  const month = Number(monthText)
  const day = Number(dayText)
  const date =
    boundary === 'start'
      ? new Date(year, month - 1, day, 0, 0, 0, 0)
      : new Date(year, month - 1, day, 23, 59, 59, 999)

  if (
    !Number.isFinite(date.getTime()) ||
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day
  ) {
    return null
  }

  return date.toISOString()
}

export function FilterPanel({
  searchMode = 'full_text',
  onSearchModeChange,
  mediaTypes,
  selectedMediaTypes,
  onMediaTypesChange,
  sortBy = 'relevance',
  onSortByChange,
  dateRange = { startDate: null, endDate: null },
  onDateRangeChange,
  exactPhrase = '',
  onExactPhraseChange,
  searchFields = DEFAULT_MEDIA_SEARCH_FIELDS,
  onSearchFieldsChange,
  enableBoostFields = false,
  onEnableBoostFieldsChange,
  boostFields = { title: 2, content: 1 },
  onBoostFieldsChange,
  metadataFilters = [createMetadataSearchFilter()],
  onMetadataFiltersChange,
  metadataMatchMode = 'all',
  onMetadataMatchModeChange,
  metadataValidationError = null,
  selectedKeywords,
  onKeywordsChange,
  selectedExcludedKeywords = [],
  onExcludedKeywordsChange,
  keywordOptions = [],
  keywordSourceMode = 'results',
  onKeywordSearch,
  showFavoritesOnly = false,
  onShowFavoritesOnlyChange,
  favoritesCount = 0,
  activeFilterCount = 0,
  onClearAll
}: FilterPanelProps) {
  const { t } = useTranslation(['review'])
  const [keywordSearchText, setKeywordSearchText] = useState('')
  const [excludeKeywordSearchText, setExcludeKeywordSearchText] = useState('')
  const [expandedSections, setExpandedSections] = useState({
    mediaTypes: false,
    advanced: false,
    keywords: false,
  })
  const [mediaTypesSectionTouched, setMediaTypesSectionTouched] = useState(false)
  const normalizedSearchFields = useMemo(
    () => normalizeMediaSearchFields(searchFields),
    [searchFields]
  )
  const normalizedMetadataFilters = useMemo(
    () => normalizeMetadataSearchFilters(metadataFilters),
    [metadataFilters]
  )
  const derivedActiveFilterCount = useMemo(() => {
    if (activeFilterCount > 0) return activeFilterCount
    return (
      selectedMediaTypes.length +
      selectedKeywords.length +
      selectedExcludedKeywords.length +
      Number(Boolean(dateRange?.startDate || dateRange?.endDate)) +
      Number(sortBy !== 'relevance') +
      Number(Boolean(exactPhrase.trim())) +
      Number(!hasDefaultMediaSearchFields(normalizedSearchFields)) +
      Number(enableBoostFields) +
      Number(searchMode === 'metadata') +
      Number(normalizedMetadataFilters.length > 0) +
      Number(showFavoritesOnly)
    )
  }, [
    activeFilterCount,
    dateRange?.endDate,
    dateRange?.startDate,
    enableBoostFields,
    exactPhrase,
    normalizedMetadataFilters.length,
    normalizedSearchFields,
    searchMode,
    selectedExcludedKeywords.length,
    selectedKeywords.length,
    selectedMediaTypes.length,
    showFavoritesOnly,
    sortBy
  ])

  const sortOptions = useMemo(
    () => [
      {
        value: 'relevance',
        label: t('review:mediaPage.sortRelevance', { defaultValue: 'Relevance' })
      },
      {
        value: 'date_desc',
        label: t('review:mediaPage.sortDateNewest', {
          defaultValue: 'Date (newest first)'
        })
      },
      {
        value: 'date_asc',
        label: t('review:mediaPage.sortDateOldest', {
          defaultValue: 'Date (oldest first)'
        })
      },
      {
        value: 'title_asc',
        label: t('review:mediaPage.sortTitleAZ', {
          defaultValue: 'Title (A-Z)'
        })
      },
      {
        value: 'title_desc',
        label: t('review:mediaPage.sortTitleZA', {
          defaultValue: 'Title (Z-A)'
        })
      }
    ],
    [t]
  )
  const keywordSuggestionCount = useMemo(() => {
    const normalizedQuery = keywordSearchText.trim().toLowerCase()
    if (!normalizedQuery) return keywordOptions.length
    return keywordOptions.filter((keyword) =>
      String(keyword).toLowerCase().includes(normalizedQuery)
    ).length
  }, [keywordOptions, keywordSearchText])

  const excludedKeywordSuggestionCount = useMemo(() => {
    const normalizedQuery = excludeKeywordSearchText.trim().toLowerCase()
    if (!normalizedQuery) return keywordOptions.length
    return keywordOptions.filter((keyword) =>
      String(keyword).toLowerCase().includes(normalizedQuery)
    ).length
  }, [excludeKeywordSearchText, keywordOptions])

  const startDateInputValue = formatDateInputValue(dateRange?.startDate)
  const endDateInputValue = formatDateInputValue(dateRange?.endDate)

  const toggleSection = (section: keyof typeof expandedSections) => {
    if (section === 'mediaTypes') {
      setMediaTypesSectionTouched(true)
    }
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  useEffect(() => {
    if (!mediaTypesSectionTouched && mediaTypes.length > 0) {
      setExpandedSections((prev) => {
        if (prev.mediaTypes) return prev
        return {
          ...prev,
          mediaTypes: true
        }
      })
    }
  }, [mediaTypes.length, mediaTypesSectionTouched])

  const handleMediaTypeToggle = (type: string) => {
    if (selectedMediaTypes.includes(type)) {
      onMediaTypesChange(selectedMediaTypes.filter(t => t !== type))
    } else {
      onMediaTypesChange([...selectedMediaTypes, type])
    }
  }

  const handleSearchFieldToggle = (field: MediaSearchField) => {
    if (!onSearchFieldsChange) return
    const hasField = normalizedSearchFields.includes(field)
    const nextFields = hasField
      ? normalizedSearchFields.filter((currentField) => currentField !== field)
      : [...normalizedSearchFields, field]

    if (nextFields.length === 0) {
      return
    }
    onSearchFieldsChange(normalizeMediaSearchFields(nextFields))
  }

  const handleBoostFieldChange = (field: keyof MediaBoostFields, value: string) => {
    const parsed = Number(value)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return
    }

    onBoostFieldsChange?.({
      ...boostFields,
      [field]: parsed
    })
  }

  const handleDateInputChange = (
    field: keyof MediaDateRange,
    value: string
  ) => {
    if (!onDateRangeChange) return

    onDateRangeChange({
      startDate:
        field === 'startDate'
          ? parseDateInputValue(value, 'start')
          : dateRange?.startDate ?? null,
      endDate:
        field === 'endDate'
          ? parseDateInputValue(value, 'end')
          : dateRange?.endDate ?? null
    })
  }

  const handleMetadataFilterChange = (
    filterId: string,
    patch: Partial<Omit<MetadataSearchFilter, 'id'>>
  ) => {
    if (!onMetadataFiltersChange) return
    onMetadataFiltersChange(
      metadataFilters.map((filter) => {
        if (filter.id !== filterId) return filter
        const nextField = (patch.field ?? filter.field) as MetadataSearchField
        const allowedOperators = getAllowedMetadataOperators(nextField)
        const nextOperator = (patch.op ?? filter.op) as MetadataSearchOperator
        return {
          ...filter,
          ...patch,
          field: nextField,
          op: allowedOperators.includes(nextOperator)
            ? nextOperator
            : allowedOperators[0]
        }
      })
    )
  }

  const handleAddMetadataFilter = () => {
    onMetadataFiltersChange?.([
      ...metadataFilters,
      createMetadataSearchFilter({ field: 'journal', op: 'icontains' })
    ])
  }

  const handleRemoveMetadataFilter = (filterId: string) => {
    if (!onMetadataFiltersChange) return
    const nextFilters = metadataFilters.filter((filter) => filter.id !== filterId)
    if (nextFilters.length === 0) {
      onMetadataFiltersChange([createMetadataSearchFilter()])
      return
    }
    onMetadataFiltersChange(nextFilters)
  }

  const handleClearAll = () => {
    if (onClearAll) {
      onClearAll()
      return
    }
    onMediaTypesChange([])
    onKeywordsChange([])
    onExcludedKeywordsChange?.([])
    onDateRangeChange?.({ startDate: null, endDate: null })
    onSortByChange?.('relevance')
    onExactPhraseChange?.('')
    onSearchFieldsChange?.([...DEFAULT_MEDIA_SEARCH_FIELDS])
    onEnableBoostFieldsChange?.(false)
    onBoostFieldsChange?.({ title: 2, content: 1 })
    onMetadataFiltersChange?.([createMetadataSearchFilter()])
    onMetadataMatchModeChange?.('all')
    onShowFavoritesOnlyChange?.(false)
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-text">
          <Filter className="w-4 h-4" />
          <span>
            {t('review:reviewPage.filters', { defaultValue: 'Filters' })}
          </span>
        </div>
        <button
          type="button"
          onClick={handleClearAll}
          className="text-sm text-primary hover:text-primaryStrong"
          title={
            derivedActiveFilterCount > 0
              ? t('review:mediaPage.clearAllWithCount', {
                  defaultValue: 'Clear all ({{count}})',
                  count: derivedActiveFilterCount
                })
              : t('review:mediaPage.clearAll', { defaultValue: 'Clear all' })
          }
        >
          {derivedActiveFilterCount > 0
            ? t('review:mediaPage.clearAllWithCount', {
                defaultValue: 'Clear all ({{count}})',
                count: derivedActiveFilterCount
              })
            : t('review:mediaPage.clearAll', { defaultValue: 'Clear all' })}
        </button>
      </div>

      <div className="space-y-2">
        <div className="text-sm text-text">
          {t('review:mediaPage.searchMode', { defaultValue: 'Search mode' })}
        </div>
        <div className="inline-flex w-full items-center gap-1 rounded-lg border border-border bg-surface2 p-1">
          <button
            type="button"
            onClick={() => onSearchModeChange?.('full_text')}
            className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
              searchMode === 'full_text'
                ? 'bg-primary text-white'
                : 'text-text hover:bg-surface'
            }`}
            aria-pressed={searchMode === 'full_text'}
            aria-label={t('review:mediaPage.searchModeFullText', {
              defaultValue: 'Full-text search'
            })}
          >
            {t('review:mediaPage.searchModeFullText', { defaultValue: 'Full-text' })}
          </button>
          <button
            type="button"
            onClick={() => onSearchModeChange?.('metadata')}
            className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
              searchMode === 'metadata'
                ? 'bg-primary text-white'
                : 'text-text hover:bg-surface'
            }`}
            aria-pressed={searchMode === 'metadata'}
            aria-label={t('review:mediaPage.searchModeMetadata', {
              defaultValue: 'Metadata search'
            })}
          >
            {t('review:mediaPage.searchModeMetadata', { defaultValue: 'Metadata' })}
          </button>
        </div>
      </div>

      {/* Sort */}
      <div className="space-y-2">
        <div className="text-sm text-text">
          {t('review:mediaPage.sortBy', { defaultValue: 'Sort by' })}
        </div>
        <Select
          value={sortBy}
          onChange={(value) => onSortByChange?.(value as MediaSortBy)}
          options={sortOptions}
          className="w-full"
        />
      </div>

      {/* Date Range */}
      <div className="space-y-2">
        <div className="text-sm text-text">
          {t('review:mediaPage.dateRange', { defaultValue: 'Date range' })}
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <label className="space-y-1 text-xs text-textMuted">
            <span>
              {t('review:mediaPage.startDate', { defaultValue: 'Start date' })}
            </span>
            <input
              type="date"
              value={startDateInputValue}
              onChange={(event) =>
                handleDateInputChange('startDate', event.currentTarget.value)
              }
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </label>
          <label className="space-y-1 text-xs text-textMuted">
            <span>
              {t('review:mediaPage.endDate', { defaultValue: 'End date' })}
            </span>
            <input
              type="date"
              value={endDateInputValue}
              onChange={(event) =>
                handleDateInputChange('endDate', event.currentTarget.value)
              }
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </label>
        </div>
      </div>

      {searchMode === 'full_text' && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => toggleSection('advanced')}
            className="flex items-center justify-between w-full text-sm text-text hover:text-text"
            title={t('review:mediaPage.advancedSearch', {
              defaultValue: 'Advanced search'
            })}
          >
            <span>
              {t('review:mediaPage.advancedSearch', {
                defaultValue: 'Advanced search'
              })}
            </span>
            <ChevronDown
              className={`w-4 h-4 transition-transform ${expandedSections.advanced ? 'rotate-180' : ''}`}
            />
          </button>

          {expandedSections.advanced && (
            <div className="space-y-3 pl-1">
              <div className="space-y-1">
                <label className="text-xs text-text-muted">
                  {t('review:mediaPage.exactPhrase', { defaultValue: 'Exact phrase' })}
                </label>
                <input
                  type="text"
                  value={exactPhrase}
                  onChange={(event) => onExactPhraseChange?.(event.target.value)}
                  placeholder={t('review:mediaPage.exactPhrasePlaceholder', {
                    defaultValue: 'e.g., "systematic review"'
                  })}
                  aria-label={t('review:mediaPage.exactPhrase', { defaultValue: 'Exact phrase' })}
                  className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-focus focus:border-transparent"
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs text-text-muted">
                  {t('review:mediaPage.searchScope', { defaultValue: 'Search scope' })}
                </label>
                <div className="flex items-center gap-3">
                  <label className="inline-flex items-center gap-2 text-sm text-text cursor-pointer">
                    <input
                      type="checkbox"
                      checked={normalizedSearchFields.includes('title')}
                      onChange={() => handleSearchFieldToggle('title')}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-focus"
                    />
                    {t('review:mediaPage.searchTitles', { defaultValue: 'Search titles' })}
                  </label>
                  <label className="inline-flex items-center gap-2 text-sm text-text cursor-pointer">
                    <input
                      type="checkbox"
                      checked={normalizedSearchFields.includes('content')}
                      onChange={() => handleSearchFieldToggle('content')}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-focus"
                    />
                    {t('review:mediaPage.searchContent', { defaultValue: 'Search content' })}
                  </label>
                </div>
              </div>

              <div className="space-y-2">
                <label className="inline-flex items-center gap-2 text-sm text-text cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableBoostFields}
                    onChange={(event) => onEnableBoostFieldsChange?.(event.target.checked)}
                    className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-focus"
                  />
                  {t('review:mediaPage.enableRelevanceBoost', {
                    defaultValue: 'Use custom relevance weights'
                  })}
                </label>
                {enableBoostFields && (
                  <div className="grid grid-cols-2 gap-2">
                    <label className="space-y-1">
                      <span className="text-xs text-text-muted">
                        {t('review:mediaPage.titleWeight', { defaultValue: 'Title weight' })}
                      </span>
                      <input
                        type="number"
                        min={0.1}
                        step={0.1}
                        value={boostFields.title ?? 2}
                        onChange={(event) =>
                          handleBoostFieldChange('title', event.target.value)
                        }
                        aria-label={t('review:mediaPage.titleWeight', { defaultValue: 'Title weight' })}
                        className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text focus:outline-none focus:ring-2 focus:ring-focus focus:border-transparent"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-xs text-text-muted">
                        {t('review:mediaPage.contentWeight', { defaultValue: 'Content weight' })}
                      </span>
                      <input
                        type="number"
                        min={0.1}
                        step={0.1}
                        value={boostFields.content ?? 1}
                        onChange={(event) =>
                          handleBoostFieldChange('content', event.target.value)
                        }
                        aria-label={t('review:mediaPage.contentWeight', { defaultValue: 'Content weight' })}
                        className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text focus:outline-none focus:ring-2 focus:ring-focus focus:border-transparent"
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {searchMode === 'metadata' && (
        <div className="space-y-3 rounded-lg border border-border bg-surface2/40 p-2">
          <div className="flex items-center justify-between">
            <div className="text-sm text-text">
              {t('review:mediaPage.metadataFilters', { defaultValue: 'Metadata filters' })}
            </div>
            <button
              type="button"
              onClick={handleAddMetadataFilter}
              className="inline-flex items-center gap-1 text-xs text-primary hover:text-primaryStrong"
            >
              <Plus className="h-3.5 w-3.5" />
              {t('review:mediaPage.addFilter', { defaultValue: 'Add filter' })}
            </button>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-text-muted">
              {t('review:mediaPage.matchMode', { defaultValue: 'Match mode' })}
            </label>
            <Select
              value={metadataMatchMode}
              className="w-full"
              onChange={(value) =>
                onMetadataMatchModeChange?.(value as MetadataMatchMode)
              }
              options={[
                {
                  value: 'all',
                  label: t('review:mediaPage.matchAll', { defaultValue: 'All filters (AND)' })
                },
                {
                  value: 'any',
                  label: t('review:mediaPage.matchAny', { defaultValue: 'Any filter (OR)' })
                }
              ]}
            />
          </div>

          <div className="space-y-2">
            {metadataFilters.map((filter) => {
              const allowedOps = getAllowedMetadataOperators(filter.field)
              return (
                <div key={filter.id} className="space-y-1 rounded-md border border-border bg-surface p-2">
                  <div className="grid grid-cols-[1fr_1fr_auto] gap-2">
                    <Select
                      value={filter.field}
                      className="w-full"
                      onChange={(value) =>
                        handleMetadataFilterChange(filter.id, {
                          field: value as MetadataSearchField
                        })
                      }
                      options={METADATA_SEARCH_FIELDS}
                    />
                    <Select
                      value={allowedOps.includes(filter.op) ? filter.op : allowedOps[0]}
                      className="w-full"
                      onChange={(value) =>
                        handleMetadataFilterChange(filter.id, {
                          op: value as MetadataSearchOperator
                        })
                      }
                      options={METADATA_SEARCH_OPERATORS.filter((option) =>
                        allowedOps.includes(option.value)
                      )}
                    />
                    <button
                      type="button"
                      onClick={() => handleRemoveMetadataFilter(filter.id)}
                      className="inline-flex h-8 w-8 items-center justify-center rounded border border-border text-text-muted hover:text-warn hover:border-warn/50"
                      aria-label={t('review:mediaPage.removeFilter', { defaultValue: 'Remove filter' })}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <input
                    type="text"
                    value={filter.value}
                    onChange={(event) =>
                      handleMetadataFilterChange(filter.id, {
                        value: event.target.value
                      })
                    }
                    placeholder={t('review:mediaPage.metadataValue', { defaultValue: 'Value' })}
                    className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-focus focus:border-transparent"
                  />
                </div>
              )
            })}
          </div>

          {normalizedMetadataFilters.length === 0 && (
            <div className="text-xs text-text-subtle">
              {t('review:mediaPage.metadataFilterHint', {
                defaultValue: 'Add at least one filter to run metadata search.'
              })}
            </div>
          )}
          {metadataValidationError && (
            <div className="text-xs text-warn">{metadataValidationError}</div>
          )}
        </div>
      )}

      {/* Favorites Toggle */}
      {onShowFavoritesOnlyChange && (
        <label className="flex items-center gap-2 cursor-pointer py-1 px-1 rounded hover:bg-surface2 transition-colors">
          <input
            type="checkbox"
            checked={showFavoritesOnly}
            onChange={(e) => onShowFavoritesOnlyChange(e.target.checked)}
            className="w-4 h-4 rounded border-border text-warn focus:ring-2 focus:ring-warn"
          />
          <Star className={`w-4 h-4 text-warn ${showFavoritesOnly ? 'fill-warn' : ''}`} />
          <span className="text-sm text-text">
            {t('review:mediaPage.favoritesOnly', { defaultValue: 'Favorites only' })}
          </span>
          {favoritesCount > 0 && (
            <span className="text-xs px-1.5 py-0.5 rounded-full bg-warn/10 text-warn font-medium">
              {favoritesCount}
            </span>
          )}
        </label>
      )}

      {/* Media Types */}
      <div className="space-y-2">
        <button
          type="button"
          onClick={() => toggleSection('mediaTypes')}
          className="flex items-center justify-between w-full text-sm text-text hover:text-text"
          title={t('review:reviewPage.mediaTypes', {
            defaultValue: 'Media types'
          })}
        >
          <span>
            {t('review:reviewPage.mediaTypes', {
              defaultValue: 'Media types'
            })}
          </span>
          <ChevronDown
            className={`w-4 h-4 transition-transform ${expandedSections.mediaTypes ? 'rotate-180' : ''}`}
          />
        </button>
        {expandedSections.mediaTypes && (
          <div className="pl-1">
            {mediaTypes.length > 0 ? (
              <div className="space-y-2">
                {mediaTypes.map(type => {
                  const displayLabel = getMediaTypeLabel(type)
                  return (
                    <label key={type} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedMediaTypes.includes(type)}
                        onChange={() => handleMediaTypeToggle(type)}
                        className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-focus"
                        aria-label={t('review:mediaPage.filterMediaType', 'Filter by {{type}}', { type: displayLabel })}
                      />
                      <span className="text-sm text-text">{displayLabel}</span>
                    </label>
                  )
                })}
              </div>
            ) : (
              <div className="text-sm text-text-muted">
                {t('review:mediaPage.noMediaTypes', {
                  defaultValue: 'No media types available'
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Keywords */}
      <div className="space-y-2">
        <button
          type="button"
          onClick={() => toggleSection('keywords')}
          className="flex items-center justify-between w-full text-sm text-text hover:text-text"
          aria-expanded={expandedSections.keywords}
          aria-controls="media-keywords-panel"
        >
          <span className="inline-flex items-center gap-2">
            <span>{t('review:reviewPage.keywords', { defaultValue: 'Keywords' })}</span>
            {selectedKeywords.length > 0 && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primaryStrong font-medium">
                {t('review:mediaPage.keywordsSelected', '{{count}} selected', { count: selectedKeywords.length })}
              </span>
            )}
          </span>
          <ChevronDown
            className={`w-4 h-4 transition-transform ${expandedSections.keywords ? 'rotate-180' : ''}`}
            aria-hidden="true"
          />
        </button>
        {expandedSections.keywords && (
          <div id="media-keywords-panel" className="space-y-2">
            <Select
              mode="tags"
              allowClear
              aria-label={t('review:mediaPage.filterByKeyword', {
                defaultValue: 'Filter by keyword'
              })}
              aria-describedby="media-keyword-helper media-keyword-status media-keyword-source-mode"
              placeholder={t('review:mediaPage.filterByKeyword', {
                defaultValue: 'Filter by keyword'
              })}
              className="w-full"
              value={selectedKeywords}
              onSearch={(txt) => {
                setKeywordSearchText(txt)
                if (onKeywordSearch) onKeywordSearch(txt)
              }}
              onChange={(vals) => {
                onKeywordsChange(vals as string[])
              }}
              options={keywordOptions.map((k) => ({ label: k, value: k }))}
            />
            <div
              id="media-keyword-helper"
              className="text-xs text-text-muted"
            >
              {t('review:mediaPage.keywordHelper', {
                defaultValue:
                  'Add keywords to narrow down results. Keywords are assigned when reviewing media.'
              })}
            </div>
            <div
              id="media-keyword-status"
              className="sr-only"
              aria-live="polite"
              aria-atomic="true"
              data-testid="keyword-suggestions-status"
            >
              {`${keywordSuggestionCount} ${t('review:mediaPage.keywordSuggestionsAvailable', {
                defaultValue: 'keyword suggestions available'
              })}`}
            </div>
            {keywordSourceMode === 'results' && (
              <div id="media-keyword-source-mode" className="text-xs text-text-subtle">
                {t('review:mediaPage.keywordSourceResultsScoped', {
                  defaultValue: 'Suggestions shown here are from current results.'
                })}
              </div>
            )}
            {keywordSourceMode !== 'results' && (
              <div id="media-keyword-source-mode" className="sr-only">
                {t('review:mediaPage.keywordSourceEndpoint', {
                  defaultValue: 'Suggestions shown here are from your full keyword list.'
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Exclude Keywords */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-sm text-text">
            {t('review:mediaPage.excludeKeywords', { defaultValue: 'Exclude keywords' })}
          </div>
          {selectedExcludedKeywords.length > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-warn/10 text-warn font-medium">
              {t('review:mediaPage.keywordsExcluded', '{{count}} excluded', { count: selectedExcludedKeywords.length })}
            </span>
          )}
        </div>
        <Select
          mode="tags"
          allowClear
          aria-label={t('review:mediaPage.excludeByKeyword', {
            defaultValue: 'Exclude keyword'
          })}
          aria-describedby="media-exclude-keyword-helper media-exclude-keyword-status"
          placeholder={t('review:mediaPage.excludeByKeyword', {
            defaultValue: 'Exclude keyword'
          })}
          className="w-full"
          value={selectedExcludedKeywords}
          onSearch={(txt) => {
            setExcludeKeywordSearchText(txt)
            if (onKeywordSearch) onKeywordSearch(txt)
          }}
          onChange={(vals) => {
            onExcludedKeywordsChange?.(vals as string[])
          }}
          options={keywordOptions.map((k) => ({ label: k, value: k }))}
        />
        <div
          id="media-exclude-keyword-helper"
          className="text-xs text-text-muted"
        >
          {t('review:mediaPage.excludeKeywordHelper', {
            defaultValue: 'Use excluded keywords to remove items containing those terms.'
          })}
        </div>
        <div
          id="media-exclude-keyword-status"
          className="sr-only"
          aria-live="polite"
          aria-atomic="true"
          data-testid="exclude-keyword-suggestions-status"
        >
          {`${excludedKeywordSuggestionCount} ${t('review:mediaPage.excludeKeywordSuggestionsAvailable', {
            defaultValue: 'exclude keyword suggestions available'
          })}`}
        </div>
      </div>
    </div>
  )
}
