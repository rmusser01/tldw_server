import React from 'react'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import MediaReviewPage from '../MediaReviewPage'

const mediaItems = Array.from({ length: 10 }).map((_, idx) => ({
  id: idx + 1,
  title: `Item ${idx + 1}`,
  snippet: `Snippet ${idx + 1}`,
  type: 'pdf',
  created_at: '2026-02-17T00:00:00.000Z'
}))

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  refetch: vi.fn(),
  messageInfo: vi.fn(),
  messageWarning: vi.fn(),
  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  clearSetting: vi.fn(),
  setHelpDismissed: vi.fn(),
  setSettingHook: vi.fn(),
  setChatMode: vi.fn(),
  setSelectedKnowledge: vi.fn(),
  setRagMediaIds: vi.fn(),
  navigate: vi.fn(),
  useVirtualizer: vi.fn(({ count }: { count: number }) => ({
    getTotalSize: () => count * 120,
    getVirtualItems: () =>
      Array.from({ length: count }).map((_, index) => ({
        index,
        start: index * 120,
        size: 120,
        key: index
      })),
    scrollToIndex: vi.fn(),
    measureElement: vi.fn()
  }))
}))

const interpolate = (template: string, values?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(values?.[key] ?? ''))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === 'string') {
        return interpolate(fallbackOrOptions, maybeOptions)
      }
      const template = fallbackOrOptions?.defaultValue || key
      return interpolate(template, fallbackOrOptions as Record<string, unknown> | undefined)
    }
  })
}))

vi.mock('react-router-dom', () => ({
  useNavigate: () => mocks.navigate
}))

vi.mock('@/hooks/useMessageOption', () => ({
  useMessageOption: () => ({
    setChatMode: mocks.setChatMode,
    setSelectedKnowledge: mocks.setSelectedKnowledge,
    setRagMediaIds: mocks.setRagMediaIds
  })
}))

vi.mock('@/hooks/useAntdMessage', () => ({
  useAntdMessage: () => ({
    info: mocks.messageInfo,
    warning: mocks.messageWarning,
    success: mocks.messageSuccess,
    error: mocks.messageError
  })
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock('@tanstack/react-query', () => ({
  keepPreviousData: {},
  useQuery: () => ({
    data: mediaItems,
    isFetching: false,
    refetch: mocks.refetch
  })
}))

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: (params: { count: number }) => mocks.useVirtualizer(params)
}))

vi.mock('@/hooks/useServerOnline', () => ({
  useServerOnline: () => true
}))

vi.mock('@/services/note-keywords', () => ({
  getNoteKeywords: vi.fn().mockResolvedValue([]),
  searchNoteKeywords: vi.fn().mockResolvedValue([])
}))

vi.mock('@plasmohq/storage/hook', () => ({
  useStorage: () => [true, mocks.setHelpDismissed, { isLoading: false }]
}))

vi.mock('@/hooks/useSetting', () => {
  const React = require('react') as typeof import('react')
  return {
    useSetting: (setting: { key: string; defaultValue: unknown }) => {
      const [value, setValue] = React.useState(setting.defaultValue)
      const setter = async (next: unknown | ((prev: unknown) => unknown)) => {
        setValue((prev: unknown) => (typeof next === 'function' ? (next as (prev: unknown) => unknown)(prev) : next))
      }
      return [value, setter, { isLoading: false }] as const
    }
  }
})

vi.mock('@/services/settings/registry', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/settings/registry')>()
  return {
    ...actual,
    getSetting: mocks.getSetting,
    setSetting: mocks.setSetting,
    clearSetting: mocks.clearSetting
  }
})

vi.mock('@/services/settings/ui-settings', () => ({
  DISCUSS_MEDIA_PROMPT_SETTING: { key: 'discussMediaPrompt', defaultValue: null },
  LAST_MEDIA_ID_SETTING: { key: 'lastMediaId', defaultValue: null },
  MEDIA_HIDE_TRANSCRIPT_TIMINGS_SETTING: { key: 'mediaHideTranscriptTimings', defaultValue: true },
  MEDIA_REVIEW_AUTO_VIEW_MODE_SETTING: { key: 'mediaReviewAutoViewMode', defaultValue: true },
  MEDIA_REVIEW_FILTERS_COLLAPSED_SETTING: { key: 'mediaReviewFiltersCollapsed', defaultValue: false },
  MEDIA_REVIEW_FOCUSED_ID_SETTING: { key: 'mediaReviewFocusedId', defaultValue: null },
  MEDIA_REVIEW_ORIENTATION_SETTING: { key: 'mediaReviewOrientation', defaultValue: 'vertical' },
  MEDIA_REVIEW_SELECTION_SETTING: { key: 'mediaReviewSelection', defaultValue: [] },
  MEDIA_REVIEW_VIEW_MODE_SETTING: { key: 'mediaReviewViewMode', defaultValue: 'spread' }
}))

vi.mock('@/components/Media/DiffViewModal', () => ({
  DiffViewModal: ({ open }: { open: boolean }) =>
    open ? <div data-testid="compare-diff-modal">diff</div> : null
}))

vi.mock('antd', async (importOriginal) => {
  const React = await import('react')
  const actual = await importOriginal<typeof import('antd')>()

  const Input = React.forwardRef<HTMLInputElement, any>(
    ({ value, onChange, onPressEnter, placeholder, ...rest }, ref) => (
      <input
        ref={ref}
        value={value || ''}
        onChange={onChange}
        placeholder={placeholder}
        onKeyDown={(event) => {
          if (event.key === 'Enter') onPressEnter?.(event)
        }}
        {...rest}
      />
    )
  )

  const Button = ({ children, onClick, disabled, icon, danger: _danger, iconPlacement: _iconPlacement, ...rest }: any) => (
    <button type="button" onClick={onClick} disabled={disabled} {...rest}>
      {icon}
      {children}
    </button>
  )

  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Spin = () => <span>loading</span>
  const Pagination = () => <div>pagination</div>
  const Empty = ({ description }: any) => <div>{description}</div>
  ;(Empty as any).PRESENTED_IMAGE_SIMPLE = null

  const Checkbox = ({ checked, onChange, children, ...rest }: any) => (
    <label>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange?.({ target: { checked: event.target.checked } })}
        {...rest}
      />
      {children}
    </label>
  )

  const SelectComponent = ({ value, onChange, options = [], children, mode, ...rest }: any) => {
    const resolvedOptions = options.length > 0
      ? options
      : React.Children.toArray(children).map((child: any) => ({
          value: child?.props?.value,
          label: child?.props?.children
        }))
    const isMultiple = mode === 'multiple' || mode === 'tags'
    const selected = isMultiple ? (Array.isArray(value) ? value : []) : (value ?? '')
    return (
      <select
        multiple={isMultiple}
        value={selected as any}
        onChange={(event) => {
          if (isMultiple) {
            const values = Array.from(event.currentTarget.selectedOptions).map((opt) => opt.value)
            onChange?.(values)
          } else {
            onChange?.(event.currentTarget.value)
          }
        }}
        aria-label={rest['aria-label']}
      >
        {resolvedOptions.map((opt: any) => (
          <option key={String(opt.value)} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    )
  }
  ;(SelectComponent as any).Option = ({ value, children }: any) => (
    <option value={value}>{children}</option>
  )

  const RadioButton = ({ value, children, __groupValue, __groupOnChange }: any) => (
    <button
      type="button"
      aria-pressed={__groupValue === value}
      onClick={() => __groupOnChange?.({ target: { value } })}
    >
      {children}
    </button>
  )

  const RadioGroup = ({ value, onChange, children }: any) => (
    <div>
      {React.Children.map(children, (child: any) =>
        React.cloneElement(child, {
          __groupValue: value,
          __groupOnChange: onChange
        })
      )}
    </div>
  )

  const Switch = ({ checked, onChange }: any) => (
    <input type="checkbox" checked={checked} onChange={(e) => onChange?.(e.target.checked)} />
  )
  const Typography = { Text: ({ children }: any) => <span>{children}</span> }
  const Skeleton = () => <div>loading-skeleton</div>
  const Alert = ({ title, action }: any) => <div>{title}{action}</div>
  const Dropdown = ({ menu, children }: any) => (
    <div>
      {children}
      <div>
        {(menu?.items || [])
          .filter((item: any) => item && item.type !== 'divider')
          .map((item: any, idx: number) => (
            <button
              key={`${item.key}-${idx}`}
              type="button"
              disabled={item.disabled}
              onClick={() => item.onClick?.()}
            >
              {typeof item.label === 'string' ? item.label : String(item.key)}
            </button>
          ))}
      </div>
    </div>
  )
  const Modal = ({ open, title, onCancel, children }: any) => {
    if (!open) return null
    return (
      <div>
        <h2>{title}</h2>
        <button type="button" onClick={onCancel}>Close</button>
        {children}
      </div>
    )
  }
  const Drawer = ({ open, title, onClose, children }: any) => {
    if (!open) return null
    return (
      <div role="dialog" aria-label={typeof title === 'string' ? title : 'drawer'}>
        <button type="button" onClick={onClose}>Close drawer</button>
        {children}
      </div>
    )
  }

  return {
    ...actual,
    Input, Button, Spin, Tag, Tooltip,
    Radio: { Group: RadioGroup, Button: RadioButton },
    Pagination, Empty, Select: SelectComponent, Checkbox,
    Typography, Skeleton, Switch, Alert,
    Collapse: ({ children }: any) => <div>{children}</div>,
    Dropdown, Modal, Drawer
  }
})

vi.mock("@/components/Common/Markdown", () => ({
  Markdown: ({
    message,
    headingAnchorIds = []
  }: {
    message: string
    headingAnchorIds?: string[]
  }) => {
    const React = require('react') as typeof import('react')
    let headingIndex = 0

    return (
      <div data-testid="mock-markdown">
        {message.split('\n').map((line, index) => {
          const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)
          if (headingMatch) {
            const level = Math.min(headingMatch[1].length, 6)
            const anchorId = headingAnchorIds[headingIndex]
            headingIndex += 1
            return React.createElement(`h${level}`, {
              key: `${line}-${index}`,
              ...(anchorId ? { "data-section-anchor": anchorId } : {})
            }, headingMatch[2])
          }

          return <p key={`${line}-${index}`}>{line}</p>
        })}
      </div>
    )
  }
}))

vi.mock("@/components/Review/ContentRenderer", () => ({
  ContentRenderer: ({
    content,
    headingAnchorIds: explicitIds
  }: {
    content: string
    headingAnchorIds?: string[]
  }) => {
    const React = require('react') as typeof import('react')

    // Auto-generate heading anchor IDs from content (mirrors real ContentRenderer behavior)
    const autoIds: string[] = []
    content.split('\n').forEach((line, i) => {
      if (/^#{1,6}\s+/.test(line)) autoIds.push(`section-${i}`)
    })
    const anchorIds = explicitIds && explicitIds.length > 0 ? explicitIds : autoIds
    let headingIndex = 0

    return (
      <div data-testid="mock-content-renderer">
        {content.split('\n').map((line, index) => {
          const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)
          if (headingMatch) {
            const level = Math.min(headingMatch[1].length, 6)
            const anchorId = anchorIds[headingIndex]
            headingIndex += 1
            return React.createElement(`h${level}`, {
              key: `${line}-${index}`,
              ...(anchorId ? { "data-section-anchor": anchorId } : {})
            }, headingMatch[2])
          }

          return <p key={`${line}-${index}`}>{line}</p>
        })}
      </div>
    )
  }
}))

vi.mock("@/components/Media/diff-worker-client", () => ({
  computeDiffSync: () => [],
  shouldUseWorkerDiff: () => false,
  shouldRequireSampling: () => false,
  sampleTextForDiff: (t: string) => t,
  computeDiffWithWorker: async () => [],
  createDiffWorker: () => null,
  DIFF_SYNC_LINE_THRESHOLD: 4000,
  DIFF_HARD_CHAR_THRESHOLD: 300_000,
  DIFF_SAMPLED_CHAR_BUDGET: 120_000
}))

describe('MediaReviewPage stage7 three-panel layout', () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.refetch.mockReset()
    mocks.messageInfo.mockReset()
    mocks.messageWarning.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.getSetting.mockReset()
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.navigate.mockReset()
    mocks.useVirtualizer.mockClear()

    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.clearSetting.mockResolvedValue(undefined)

    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      const idMatch = path.match(/\/api\/v1\/media\/([^?]+)/)
      const id = idMatch ? Number(idMatch[1]) : null
      return {
        id: id ?? 1,
        title: `Item ${id ?? 1}`,
        type: 'pdf',
        content: `Content for item ${id ?? 1}`
      }
    })

    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      configurable: true,
      value: vi.fn().mockImplementation(() => ({
        matches: false,
        media: '',
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
  })

  const getResultRowByTitle = (title: string): HTMLElement => {
    const list = screen.getByTestId('media-review-results-list')
    const text = within(list).getByText(title)
    const row = text.closest('[role="button"]')
    if (!row) throw new Error(`Missing row for title: ${title}`)
    return row as HTMLElement
  }

  const selectItemByCheckbox = (title: string, options?: Record<string, unknown>): void => {
    const row = getResultRowByTitle(title)
    const checkbox = within(row).getByRole('checkbox')
    fireEvent.click(checkbox, options)
  }

  it('renders three-panel layout with filter, results, and reading pane', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('resizable-panels')).toBeInTheDocument()
    })

    // Left panel: filter sidebar
    expect(screen.getByTestId('panel-left')).toBeInTheDocument()
    // Center panel: results
    expect(screen.getByTestId('panel-center')).toBeInTheDocument()
    // Right panel: reading pane
    expect(screen.getByTestId('panel-right')).toBeInTheDocument()
  })

  it('renders resizable drag handles between panels', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('resizable-panels')).toBeInTheDocument()
    })

    const separators = screen.getAllByRole('separator')
    expect(separators).toHaveLength(2)
    expect(separators[0]).toHaveAttribute('aria-label', 'Resize filter panel')
    expect(separators[1]).toHaveAttribute('aria-label', 'Resize results panel')
  })

  it('clicking a result row previews without selecting', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // Click the row (not checkbox) - should preview, not select
    fireEvent.click(getResultRowByTitle('Item 1'))

    await waitFor(() => {
      // Selection count should remain 0
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // The row should be marked as previewed (aria-current)
    const row = getResultRowByTitle('Item 1')
    expect(row).toHaveAttribute('aria-current', 'true')
  })

  it('clicking a checkbox toggles selection without affecting preview', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // Select via checkbox
    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    // The row's checkbox should be checked
    const row = getResultRowByTitle('Item 1')
    const checkbox = within(row).getByRole('checkbox') as HTMLInputElement
    expect(checkbox.checked).toBe(true)

    // Deselect via checkbox
    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })
  })

  it('preview and selection are independent', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // Preview Item 1 (click row)
    fireEvent.click(getResultRowByTitle('Item 1'))

    // Select Item 2 (click checkbox)
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      // Only 1 item selected (Item 2), but Item 1 is previewed
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    // Item 1 is previewed (aria-current)
    expect(getResultRowByTitle('Item 1')).toHaveAttribute('aria-current', 'true')
    // Item 2 is selected (aria-selected)
    expect(getResultRowByTitle('Item 2')).toHaveAttribute('aria-selected', 'true')
    // Item 1 is NOT selected
    expect(getResultRowByTitle('Item 1')).toHaveAttribute('aria-selected', 'false')
  })

  it('restores persisted string ids as selected result rows', async () => {
    mocks.getSetting.mockImplementation(async (setting: { key?: string } | null) => {
      if (setting?.key === 'mediaReviewSelection') return ['2']
      return null
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    expect(getResultRowByTitle('Item 2')).toHaveAttribute('aria-selected', 'true')
    expect(getResultRowByTitle('Item 1')).toHaveAttribute('aria-selected', 'false')
  })

  it('shift+click on checkbox performs range selection', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 5', { shiftKey: true })

    await waitFor(() => {
      expect(screen.getByText('5 / 30 selected')).toBeInTheDocument()
    })
  })

  it('treats restored string ids as the same selection as numeric result ids', async () => {
    mocks.getSetting.mockImplementation(async (setting: { key?: string }) => {
      if (setting?.key === 'mediaReviewSelection') return ['1']
      return null
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    const row = getResultRowByTitle('Item 1')
    expect(row).toHaveAttribute('aria-selected', 'true')
    expect(within(row).getByRole('checkbox')).toBeChecked()

    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })
  })

  it('j/k keyboard navigation moves preview cursor through results', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // First click to preview Item 1 so we have a starting position
    fireEvent.click(getResultRowByTitle('Item 1'))

    await waitFor(() => {
      expect(getResultRowByTitle('Item 1')).toHaveAttribute('aria-current', 'true')
    })

    // Navigate down with j (from Item 1 to Item 2)
    fireEvent.keyDown(document, { key: 'j' })

    await waitFor(() => {
      const row2 = getResultRowByTitle('Item 2')
      expect(row2).toHaveAttribute('aria-current', 'true')
    })

    // Navigate up with k (from Item 2 back to Item 1)
    fireEvent.keyDown(document, { key: 'k' })

    await waitFor(() => {
      const row1 = getResultRowByTitle('Item 1')
      expect(row1).toHaveAttribute('aria-current', 'true')
    })

    // Selection should still be empty throughout
    expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
  })

  it('x key toggles selection on previewed/focused item', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // Preview Item 1
    fireEvent.click(getResultRowByTitle('Item 1'))

    // Press x to select it
    fireEvent.keyDown(document, { key: 'x' })

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    // Press x again to deselect
    fireEvent.keyDown(document, { key: 'x' })

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })
  })

  it('batch bar appears at bottom of results panel when items are selected', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    // No batch bar initially
    expect(screen.queryByTestId('media-multi-batch-toolbar')).not.toBeInTheDocument()

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByTestId('media-multi-batch-toolbar')).toBeInTheDocument()
    })

    // Batch bar should be inside the center panel
    const centerPanel = screen.getByTestId('panel-center')
    expect(within(centerPanel).getByTestId('media-multi-batch-toolbar')).toBeInTheDocument()
  })

  it('filter sidebar is always visible in left panel', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('panel-left')).toBeInTheDocument()
    })

    // Search input should be in the left panel
    const leftPanel = screen.getByTestId('panel-left')
    expect(within(leftPanel).getByLabelText(/search media/i)).toBeInTheDocument()

    // Sort control should be visible (in filter sidebar)
    expect(screen.getByRole('combobox', { name: /sort/i })).toBeInTheDocument()
  })

  it('collapses to stacked layout on mobile viewports', async () => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === '(max-width: 1023px)' ? true : false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('resizable-panels-collapsed')).toBeInTheDocument()
    })

    // In collapsed mode, no drag separators
    expect(screen.queryAllByRole('separator')).toHaveLength(0)
  })

  it('Enter/Space on result row toggles selection for keyboard a11y', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    const row = getResultRowByTitle('Item 1')
    fireEvent.keyDown(row, { key: 'Enter' })

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })
  })
})
