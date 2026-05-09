import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { FilterPanel } from '../FilterPanel'

const isoForLocalDate = (
  value: string,
  boundary: 'start' | 'end'
): string => {
  const [year, month, day] = value.split('-').map(Number)
  const date =
    boundary === 'start'
      ? new Date(year, month - 1, day, 0, 0, 0, 0)
      : new Date(year, month - 1, day, 23, 59, 59, 999)
  return date.toISOString()
}

const renderFilterPanel = (
  props: Partial<ComponentProps<typeof FilterPanel>> = {}
) =>
  render(
    <FilterPanel
      mediaTypes={[]}
      selectedMediaTypes={[]}
      onMediaTypesChange={vi.fn()}
      selectedKeywords={[]}
      onKeywordsChange={vi.fn()}
      {...props}
    />
  )

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

describe('FilterPanel', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders stage-1 controls for sort, date range, and exclude keywords', () => {
    render(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
        selectedExcludedKeywords={[]}
        onExcludedKeywordsChange={vi.fn()}
      />
    )

    expect(screen.getByText('Sort by')).toBeInTheDocument()
    expect(screen.getByText('Date range')).toBeInTheDocument()
    expect(screen.getByText('Exclude keywords')).toBeInTheDocument()
  })

  it('renders existing date range values in native date inputs', () => {
    renderFilterPanel({
      dateRange: {
        startDate: '2026-01-02T12:00:00.000Z',
        endDate: '2026-01-31T12:00:00.000Z'
      },
      onDateRangeChange: vi.fn()
    })

    const startDateInput = screen.getByLabelText('Start date') as HTMLInputElement
    const endDateInput = screen.getByLabelText('End date') as HTMLInputElement

    expect(startDateInput).toHaveAttribute('type', 'date')
    expect(startDateInput).toHaveValue('2026-01-02')
    expect(endDateInput).toHaveAttribute('type', 'date')
    expect(endDateInput).toHaveValue('2026-01-31')
  })

  it('emits local day-boundary ISO values from native date input changes', () => {
    const onDateRangeChange = vi.fn()
    const existingStartDate = '2026-01-01T00:00:00.000Z'
    const existingEndDate = '2026-01-31T23:59:59.999Z'

    renderFilterPanel({
      dateRange: {
        startDate: existingStartDate,
        endDate: existingEndDate
      },
      onDateRangeChange
    })

    fireEvent.change(screen.getByLabelText('Start date'), {
      target: { value: '2026-02-03' }
    })
    fireEvent.change(screen.getByLabelText('End date'), {
      target: { value: '2026-02-04' }
    })

    expect(onDateRangeChange).toHaveBeenNthCalledWith(1, {
      startDate: isoForLocalDate('2026-02-03', 'start'),
      endDate: existingEndDate
    })
    expect(onDateRangeChange).toHaveBeenNthCalledWith(2, {
      startDate: existingStartDate,
      endDate: isoForLocalDate('2026-02-04', 'end')
    })
  })

  it('clears native date inputs independently and preserves the other boundary', () => {
    const onDateRangeChange = vi.fn()
    const existingStartDate = '2026-01-01T00:00:00.000Z'
    const existingEndDate = '2026-01-31T23:59:59.999Z'
    const { rerender } = renderFilterPanel({
      dateRange: {
        startDate: existingStartDate,
        endDate: existingEndDate
      },
      onDateRangeChange
    })

    fireEvent.change(screen.getByLabelText('Start date'), {
      target: { value: '' }
    })
    expect(onDateRangeChange).toHaveBeenLastCalledWith({
      startDate: null,
      endDate: existingEndDate
    })

    onDateRangeChange.mockClear()
    rerender(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
        dateRange={{
          startDate: existingStartDate,
          endDate: existingEndDate
        }}
        onDateRangeChange={onDateRangeChange}
      />
    )
    fireEvent.change(screen.getByLabelText('End date'), {
      target: { value: '' }
    })
    expect(onDateRangeChange).toHaveBeenLastCalledWith({
      startDate: existingStartDate,
      endDate: null
    })

    onDateRangeChange.mockClear()
    rerender(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
        dateRange={{
          startDate: null,
          endDate: existingEndDate
        }}
        onDateRangeChange={onDateRangeChange}
      />
    )
    fireEvent.change(screen.getByLabelText('End date'), {
      target: { value: '' }
    })
    expect(onDateRangeChange).toHaveBeenLastCalledWith({
      startDate: null,
      endDate: null
    })
  })

  it('clear all resets include/exclude keywords, date range, and sort state', () => {
    const onMediaTypesChange = vi.fn()
    const onKeywordsChange = vi.fn()
    const onExcludedKeywordsChange = vi.fn()
    const onDateRangeChange = vi.fn()
    const onSortByChange = vi.fn()
    const onExactPhraseChange = vi.fn()
    const onSearchFieldsChange = vi.fn()
    const onEnableBoostFieldsChange = vi.fn()
    const onBoostFieldsChange = vi.fn()
    const onMetadataFiltersChange = vi.fn()
    const onMetadataMatchModeChange = vi.fn()
    const onShowFavoritesOnlyChange = vi.fn()

    render(
      <FilterPanel
        mediaTypes={['pdf']}
        selectedMediaTypes={['pdf']}
        onMediaTypesChange={onMediaTypesChange}
        sortBy="date_desc"
        onSortByChange={onSortByChange}
        dateRange={{
          startDate: '2026-01-01T00:00:00.000Z',
          endDate: '2026-01-31T23:59:59.999Z'
        }}
        onDateRangeChange={onDateRangeChange}
        exactPhrase="systematic review"
        onExactPhraseChange={onExactPhraseChange}
        searchFields={['title']}
        onSearchFieldsChange={onSearchFieldsChange}
        enableBoostFields
        onEnableBoostFieldsChange={onEnableBoostFieldsChange}
        boostFields={{ title: 3, content: 0.8 }}
        onBoostFieldsChange={onBoostFieldsChange}
        metadataFilters={[
          { id: '1', field: 'doi', op: 'eq', value: '10.1000/xyz' }
        ]}
        onMetadataFiltersChange={onMetadataFiltersChange}
        metadataMatchMode="any"
        onMetadataMatchModeChange={onMetadataMatchModeChange}
        selectedKeywords={['include-me']}
        onKeywordsChange={onKeywordsChange}
        selectedExcludedKeywords={['exclude-me']}
        onExcludedKeywordsChange={onExcludedKeywordsChange}
        showFavoritesOnly={true}
        onShowFavoritesOnlyChange={onShowFavoritesOnlyChange}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /Clear all/ }))

    expect(onMediaTypesChange).toHaveBeenCalledWith([])
    expect(onKeywordsChange).toHaveBeenCalledWith([])
    expect(onExcludedKeywordsChange).toHaveBeenCalledWith([])
    expect(onDateRangeChange).toHaveBeenCalledWith({ startDate: null, endDate: null })
    expect(onSortByChange).toHaveBeenCalledWith('relevance')
    expect(onExactPhraseChange).toHaveBeenCalledWith('')
    expect(onSearchFieldsChange).toHaveBeenCalledWith(['title', 'content'])
    expect(onEnableBoostFieldsChange).toHaveBeenCalledWith(false)
    expect(onBoostFieldsChange).toHaveBeenCalledWith({ title: 2, content: 1 })
    expect(onMetadataMatchModeChange).toHaveBeenCalledWith('all')
    expect(onMetadataFiltersChange).toHaveBeenCalled()
    expect(onShowFavoritesOnlyChange).toHaveBeenCalledWith(false)
  })

  it('supports advanced search controls for exact phrase, scope, and relevance boost', () => {
    const onExactPhraseChange = vi.fn()
    const onSearchFieldsChange = vi.fn()
    const onEnableBoostFieldsChange = vi.fn()
    const onBoostFieldsChange = vi.fn()

    render(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        exactPhrase=""
        onExactPhraseChange={onExactPhraseChange}
        searchFields={['title', 'content']}
        onSearchFieldsChange={onSearchFieldsChange}
        enableBoostFields
        onEnableBoostFieldsChange={onEnableBoostFieldsChange}
        boostFields={{ title: 2, content: 1 }}
        onBoostFieldsChange={onBoostFieldsChange}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Advanced search' }))
    fireEvent.change(screen.getByLabelText('Exact phrase'), {
      target: { value: 'randomized controlled trial' }
    })
    fireEvent.click(screen.getByLabelText('Search content'))
    fireEvent.change(screen.getByLabelText('Title weight'), {
      target: { value: '2.5' }
    })

    expect(onExactPhraseChange).toHaveBeenCalledWith('randomized controlled trial')
    expect(onSearchFieldsChange).toHaveBeenCalledWith(['title'])
    expect(onBoostFieldsChange).toHaveBeenCalledWith({ title: 2.5, content: 1 })
  })

  it('auto-expands media types section when media types are available', async () => {
    render(
      <FilterPanel
        mediaTypes={['pdf']}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('PDF')).toBeInTheDocument()
    })
  })

  it('renders metadata mode controls and supports adding filters', () => {
    const onMetadataFiltersChange = vi.fn()

    render(
      <FilterPanel
        searchMode="metadata"
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        metadataFilters={[
          { id: '1', field: 'doi', op: 'eq', value: '10.1000/xyz' }
        ]}
        onMetadataFiltersChange={onMetadataFiltersChange}
        metadataMatchMode="all"
        onMetadataMatchModeChange={vi.fn()}
        metadataValidationError='DOI requires the "equals" operator.'
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
      />
    )

    expect(screen.getByText('Metadata filters')).toBeInTheDocument()
    expect(screen.getByText('DOI requires the "equals" operator.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Add filter' }))
    expect(onMetadataFiltersChange).toHaveBeenCalled()
  })

  it('shows result-scoped keyword source helper when endpoint is unavailable', () => {
    render(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
        keywordSourceMode="results"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Keywords' }))
    expect(
      screen.getByText('Suggestions shown here are from current results.')
    ).toBeInTheDocument()
  })

  it('adds accessible descriptions and live suggestion counts for keyword selects', async () => {
    const onKeywordSearch = vi.fn()

    render(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
        selectedExcludedKeywords={[]}
        onExcludedKeywordsChange={vi.fn()}
        keywordOptions={['alpha', 'beta', 'gamma']}
        onKeywordSearch={onKeywordSearch}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Keywords' }))
    expect(screen.getByTestId('keyword-suggestions-status')).toHaveTextContent(
      '3 keyword suggestions available'
    )
    expect(screen.getByTestId('exclude-keyword-suggestions-status')).toHaveTextContent(
      '3 exclude keyword suggestions available'
    )

    const includeInput = screen.getByLabelText('Filter by keyword')
    expect(includeInput).toHaveAttribute('aria-describedby')
    expect(includeInput.getAttribute('aria-describedby')).toContain('media-keyword-helper')
    expect(includeInput.getAttribute('aria-describedby')).toContain('media-keyword-status')

    const excludeInput = screen.getByLabelText('Exclude keyword')
    expect(excludeInput).toHaveAttribute('aria-describedby')
    expect(excludeInput.getAttribute('aria-describedby')).toContain(
      'media-exclude-keyword-helper'
    )
    expect(excludeInput.getAttribute('aria-describedby')).toContain(
      'media-exclude-keyword-status'
    )

    fireEvent.change(includeInput, { target: { value: 'ga' } })
    fireEvent.change(excludeInput, { target: { value: 'al' } })

    await waitFor(() => {
      expect(onKeywordSearch).toHaveBeenCalledWith('ga')
      expect(onKeywordSearch).toHaveBeenCalledWith('al')
    })
    expect(screen.getByTestId('keyword-suggestions-status')).toHaveTextContent(
      '1 keyword suggestions available'
    )
    expect(screen.getByTestId('exclude-keyword-suggestions-status')).toHaveTextContent(
      '1 exclude keyword suggestions available'
    )
  })

  it('allows collapsing and expanding the keywords section', () => {
    render(
      <FilterPanel
        mediaTypes={[]}
        selectedMediaTypes={[]}
        onMediaTypesChange={vi.fn()}
        selectedKeywords={[]}
        onKeywordsChange={vi.fn()}
      />
    )

    const keywordsToggle = screen.getByRole('button', { name: 'Keywords' })
    expect(keywordsToggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText('Filter by keyword')).not.toBeInTheDocument()

    fireEvent.click(keywordsToggle)
    expect(keywordsToggle).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByLabelText('Filter by keyword')).toBeInTheDocument()

    fireEvent.click(keywordsToggle)
    expect(keywordsToggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText('Filter by keyword')).not.toBeInTheDocument()
  })
})
