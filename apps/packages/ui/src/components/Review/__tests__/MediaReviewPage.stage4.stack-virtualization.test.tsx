import React from 'react'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import MediaReviewPage from '../MediaReviewPage'
import { STACK_VIRTUAL_ESTIMATE_SIZE } from '../stack-virtualization'

const mediaItems = Array.from({ length: 31 }).map((_, idx) => ({
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
  setChatMode: vi.fn(),
  setSelectedKnowledge: vi.fn(),
  setRagMediaIds: vi.fn(),
  navigate: vi.fn(),
  useVirtualizer: vi.fn(({ count, estimateSize }: { count: number; estimateSize?: () => number }) => {
    const estimatedSize = typeof estimateSize === 'function' ? estimateSize() : 120
    const shouldTrim = estimatedSize >= STACK_VIRTUAL_ESTIMATE_SIZE && count >= 30
    const renderedCount = shouldTrim ? 8 : count
    return {
      getTotalSize: () => count * estimatedSize,
      getVirtualItems: () =>
        Array.from({ length: renderedCount }).map((_, index) => ({
          index,
          start: index * estimatedSize,
          size: estimatedSize,
          key: index
        })),
      scrollToIndex: vi.fn(),
      measureElement: vi.fn()
    }
  })
}))

const interpolate = (template: string, values?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(values?.[key] ?? ''))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; selected?: number; limit?: number; count?: number; total?: number; current?: number },
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
  useVirtualizer: (params: { count: number; estimateSize?: () => number }) => mocks.useVirtualizer(params)
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
    useSetting: (setting: { defaultValue: unknown }) => {
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

vi.mock('@/utils/media-detail-content', async (importOriginal) => ({
  // Keep real exports (e.g. extractMediaDetailAnalysis); override only what this test controls.
  ...(await importOriginal<typeof import('@/utils/media-detail-content')>()),
  extractMediaDetailContent: (detail: any) => detail?.content || detail?.text || ''
}))

vi.mock('@/components/Media/DiffViewModal', () => ({
  DiffViewModal: () => null
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

  const Button = ({ children, onClick, disabled, icon, iconPlacement: _iconPlacement, danger: _danger, ...rest }: any) => (
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

  return {
    ...actual,
    Input,
    Button,
    Spin,
    Tag,
    Tooltip,
    Radio: { Group: RadioGroup, Button: RadioButton },
    Pagination,
    Empty,
    Select: SelectComponent,
    Checkbox,
    Typography: { Text: ({ children }: any) => <span>{children}</span> },
    Skeleton: () => <div>loading-skeleton</div>,
    Switch: ({ checked, onChange }: any) => (
      <input type="checkbox" checked={checked} onChange={(event) => onChange?.(event.target.checked)} />
    ),
    Alert: ({ title, action }: any) => (
      <div>
        {title}
        {action}
      </div>
    ),
    Collapse: ({ children }: any) => <div>{children}</div>,
    Dropdown: ({ menu, children }: any) => (
      <div>
        {children}
        <div>
          {Array.isArray(menu?.items)
            ? menu.items
                .filter((item: any) => item && item.type !== 'divider')
                .map((item: any, idx: number) => (
                  <button key={`${item.key}-${idx}`} type="button" disabled={item.disabled} onClick={() => item.onClick?.()}>
                    {typeof item.label === 'string' ? item.label : String(item.key)}
                  </button>
                ))
            : null}
        </div>
      </div>
    ),
    Modal: () => null,
    Drawer: () => null
  }
})

vi.mock("@/components/Common/Markdown", () => ({
  Markdown: ({ message }: { message: string }) => <div data-testid="mock-markdown">{message}</div>
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

describe('MediaReviewPage stage4 stack virtualization', () => {
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
        content: `Content ${id ?? 1}`
      }
    })

    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query === '(prefers-reduced-motion: reduce)' ? false : false,
        media: query,
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

  it('virtualizes stack mode cards for large selections instead of eager rendering all cards', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    for (let i = 1; i <= 30; i += 1) {
      selectItemByCheckbox(`Item ${i}`)
    }

    await waitFor(() => {
      expect(screen.getByText('30 / 30 selected')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /^stack/i }))

    await waitFor(() => {
      const copyButtons = screen.getAllByRole('button', { name: 'Copy Content' })
      expect(copyButtons.length).toBeLessThan(30)
    })

    expect(screen.getByTestId('media-review-stack-virtualized')).toBeInTheDocument()
    expect(
      mocks.useVirtualizer.mock.calls.some((call) => {
        const args = call[0] as { count?: number; estimateSize?: () => number }
        return args.count === 30 && args.estimateSize?.() === STACK_VIRTUAL_ESTIMATE_SIZE
      })
    ).toBe(true)
  })
})
