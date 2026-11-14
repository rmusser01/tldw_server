import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

import { ResultsList, type Result } from '../ResultsList'
import { buildInspectorTooltip } from '../resultInspectorTooltip'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

// Spy on the extracted tooltip builder so the test observes exactly when a
// row recomputes its inspector tooltip (cross-module import: interceptable,
// unlike a same-module internal helper).
vi.mock('../resultInspectorTooltip', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../resultInspectorTooltip')>()
  return {
    ...actual,
    buildInspectorTooltip: vi.fn(actual.buildInspectorTooltip)
  }
})

const buildResults = (count: number): Result[] =>
  Array.from({ length: count }, (_unused, index) => ({
    id: `r-${index}`,
    kind: 'media' as const,
    title: `Result ${index}`,
    snippet: `snippet ${index}`,
    meta: {
      type: 'pdf',
      author: `Author ${index}`,
      created_at: '2026-02-16T12:00:00.000Z',
      page_count: index + 1
    }
  }))

describe('ResultsList memoized rows (WP-11)', () => {
  it('does not rebuild the inspector tooltip for unchanged rows on parent re-render', () => {
    const results = buildResults(3)
    const onSelect = vi.fn()

    const view = render(
      <ResultsList
        results={results}
        selectedId={null}
        onSelect={onSelect}
        totalCount={3}
        loadedCount={3}
      />
    )

    // One tooltip build per row on initial render.
    expect(buildInspectorTooltip).toHaveBeenCalledTimes(3)
    expect(screen.getByText('Result 1')).toBeInTheDocument()

    // Parent re-renders with an unrelated prop change (counts update): rows
    // are memoized, so no tooltip builder re-invocation.
    view.rerender(
      <ResultsList
        results={results}
        selectedId={null}
        onSelect={onSelect}
        totalCount={42}
        loadedCount={3}
      />
    )

    // The parent really re-rendered: the header counts changed.
    expect(screen.getByText('3 / 42')).toBeInTheDocument()
    expect(buildInspectorTooltip).toHaveBeenCalledTimes(3)

    // Replacing one result object re-renders exactly that row: only one new
    // tooltip build, for the changed row.
    const nextResults = results.slice()
    nextResults[1] = { ...results[1], title: 'Result 1 (edited)' }

    view.rerender(
      <ResultsList
        results={nextResults}
        selectedId={null}
        onSelect={onSelect}
        totalCount={42}
        loadedCount={3}
      />
    )

    expect(screen.getByText('Result 1 (edited)')).toBeInTheDocument()
    expect(buildInspectorTooltip).toHaveBeenCalledTimes(4)
    expect(buildInspectorTooltip).toHaveBeenLastCalledWith(
      nextResults[1],
      expect.any(Function),
      undefined
    )
  })
})
