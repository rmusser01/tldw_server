import * as mediaHandoff from "@/services/tldw/media-chat-handoff"
import React from 'react'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axe from 'axe-core'
import MediaReviewPage from '../MediaReviewPage'

const mediaItems = Array.from({ length: 31 }).map((_, idx) => ({
  id: idx + 1,
  title: `Item ${idx + 1}`,
  snippet: `Snippet ${idx + 1}`,
  type: 'pdf',
  created_at: '2026-02-17T00:00:00.000Z'
}))

vi.mock('@/hooks/useHomeMilestoneScope', () => ({ useHomeMilestoneScope: () => 'server:alice' }))

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  refetch: vi.fn(),
  messageInfo: vi.fn(),
  messageWarning: vi.fn(),
  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  clipboardWriteText: vi.fn(),
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
  DiffViewModal: ({
    open,
    leftText,
    rightText,
    leftLabel,
    rightLabel,
    onClose
  }: {
    open: boolean
    leftText: string
    rightText: string
    leftLabel: string
    rightLabel: string
    onClose: () => void
  }) =>
    open ? (
      <div data-testid="compare-diff-modal">
        <div>{leftLabel}</div>
        <div>{rightLabel}</div>
        <div>{leftText}</div>
        <div>{rightText}</div>
        <button type="button" onClick={onClose}>
          Close diff
        </button>
      </div>
    ) : null
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

  const Typography = {
    Text: ({ children }: any) => <span>{children}</span>
  }

  const Skeleton = () => <div>loading-skeleton</div>

  const Alert = ({ title, action }: any) => (
    <div>
      {title}
      {action}
    </div>
  )

  const Dropdown = ({ menu, children }: any) => (
    <div>
      {children}
      <div>
        {Array.isArray(menu?.items)
          ? menu.items
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
              ))
          : null}
      </div>
    </div>
  )

  const Modal = ({ open, title, onCancel, footer, children }: any) => {
    if (!open) return null
    return (
      <div>
        <h2>{title}</h2>
        <button type="button" onClick={onCancel}>
          Close
        </button>
        {children}
        {footer}
      </div>
    )
  }

  const Drawer = ({ open, title, onClose, children }: any) => {
    if (!open) return null
    return (
      <div role="dialog" aria-label={typeof title === 'string' ? title : 'Selected items'}>
        <h2>{title}</h2>
        <button type="button" onClick={onClose}>
          Close drawer
        </button>
        {children}
      </div>
    )
  }

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
    Typography,
    Skeleton,
    Switch,
    Alert,
    Collapse: ({ children }: any) => <div>{children}</div>,
    Dropdown,
    Modal,
    Drawer
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

describe('MediaReviewPage stage 1 selection limit clarity', () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.refetch.mockReset()
    mocks.messageInfo.mockReset()
    mocks.messageWarning.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.clipboardWriteText.mockReset()
    mocks.getSetting.mockReset()
    mocks.setSetting.mockReset()
    mocks.clearSetting.mockReset()
    mocks.setHelpDismissed.mockReset()
    mocks.setChatMode.mockReset()
    mocks.setSelectedKnowledge.mockReset()
    mocks.setRagMediaIds.mockReset()
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
        content: `Content ${id ?? 1}`
      }
    })

    mocks.clipboardWriteText.mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: mocks.clipboardWriteText
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

  const setMobileViewport = (isMobile: boolean) => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      configurable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches:
          query === '(prefers-reduced-motion: reduce)'
            ? false
            : query === '(max-width: 1023px)'
              ? isMobile
              : false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
  }

  it('renders X / 30 selected and updates counter correctly on shift-click range selection', async () => {
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

  it('keeps warning/error threshold behavior predictable as selection nears and hits limit', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    for (let i = 1; i <= 25; i++) {
      selectItemByCheckbox(`Item ${i}`)
    }

    await waitFor(() => {
      expect(screen.getByText('25 / 30 selected')).toBeInTheDocument()
      expect(screen.getByText('(5 left)')).toBeInTheDocument()
    })

    for (let i = 26; i <= 30; i++) {
      selectItemByCheckbox(`Item ${i}`)
    }

    await waitFor(() => {
      expect(screen.getByText('30 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 31')

    expect(mocks.messageWarning).toHaveBeenCalledWith('Selection limit reached (30 items)')
    expect(screen.getByText('30 / 30 selected')).toBeInTheDocument()
  })

  it('supports keyboard selection and keeps counter in sync', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    const row = getResultRowByTitle('Item 2')
    row.focus()
    fireEvent.keyDown(row, { key: 'Enter' })

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    const checkbox = within(row).getByRole('checkbox')
    expect(checkbox).toBeChecked()
  })

  it('shows status bar with selection count', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    expect(screen.getByTestId('media-review-status-bar')).toBeInTheDocument()
    expect(screen.getByText('No selection')).toBeInTheDocument()

    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      const statusBar = screen.getByTestId('media-review-status-bar')
      expect(within(statusBar).getByText('1 selected')).toBeInTheDocument()
    })
  })

  it('focuses the search field when slash shortcut is pressed outside typing contexts', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText('Search media (title/content)')
    expect(searchInput).not.toHaveFocus()

    fireEvent.keyDown(document, { key: '/' })

    await waitFor(() => {
      expect(searchInput).toHaveFocus()
    })
  })

  it('shows compare content action only when exactly two items are selected', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })
    expect(screen.queryByRole('button', { name: 'Compare content' })).not.toBeInTheDocument()

    selectItemByCheckbox('Item 1')
    expect(screen.queryByRole('button', { name: 'Compare content' })).not.toBeInTheDocument()

    selectItemByCheckbox('Item 2')
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Compare content' })).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 2')
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Compare content' })).not.toBeInTheDocument()
    })
  }, 10000)

  it('opens inline comparison split with selected item content when compare action is triggered', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Compare content' })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Compare content' }))

    await waitFor(() => {
      const split = screen.getByTestId('comparison-split')
      expect(split).toBeInTheDocument()
      expect(within(split).getByText('Item 1')).toBeInTheDocument()
      expect(within(split).getByText('Item 2')).toBeInTheDocument()
    })
  })

  it('exits inline comparison when button is clicked again', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Compare content' })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Compare content' }))

    await waitFor(() => {
      expect(screen.getByTestId('comparison-split')).toBeInTheDocument()
    })

    // Button text should now say "Exit compare"
    fireEvent.click(screen.getByRole('button', { name: 'Exit compare' }))

    await waitFor(() => {
      expect(screen.queryByTestId('comparison-split')).not.toBeInTheDocument()
    })
  })

  it('shows chat-about-selection action only when at least one item is selected', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })
    expect(
      screen.queryByRole('button', { name: 'Chat about selection (1)' })
    ).not.toBeInTheDocument()

    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'Chat about selection (1)' })
      ).toBeInTheDocument()
    })
  })

  it('addresses selected media to the intended Chat route without broadcasting', async () => {
    const dispatchSpy = vi.spyOn(window, 'dispatchEvent')
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'Chat about selection (2)' })
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Chat about selection (2)' }))

    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith(expect.stringContaining('/chat?media_handoff=')))
    const route = mocks.navigate.mock.calls.at(-1)![0]
    const token = new URL(route, 'http://localhost').searchParams.get(mediaHandoff.MEDIA_CHAT_HANDOFF_PARAM)!
    expect(await mediaHandoff.readMediaChatHandoff(token, 'server:alice')).toEqual({ ownerScope: 'server:alice', mediaId: '1', mode: 'rag_media', mediaIds: [1, 2] })
    expect(dispatchSpy.mock.calls.some(([event]) => event.type === 'tldw:discuss-media')).toBe(false)
    expect(mocks.setChatMode).not.toHaveBeenCalled()
    expect(mocks.setRagMediaIds).not.toHaveBeenCalled()
    dispatchSpy.mockRestore()
  })

  it('preserves auto view-mode thresholds for 1, 2-4, and 5+ selections', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    await waitFor(() => {
      expect(screen.getByText('Single item view')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 2')
    await waitFor(() => {
      expect(screen.getByText('2 open')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 3')
    selectItemByCheckbox('Item 4')
    selectItemByCheckbox('Item 5')
    await waitFor(() => {
      expect(screen.getByText('All items (stacked)')).toBeInTheDocument()
    })
  })

  it('keeps help modal trigger and keyboard j/k navigation behavior available', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    fireEvent.click(
      screen.getByRole('button', {
        name: 'Multi-Item Review keyboard shortcuts'
      })
    )
    expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.queryByText('Keyboard Shortcuts')).not.toBeInTheDocument()

    fireEvent.click(getResultRowByTitle('Item 1'))
    await waitFor(() => {
      expect(screen.getByText('Item 1 of 31')).toBeInTheDocument()
    })

    fireEvent.keyDown(document, { key: 'j' })
    await waitFor(() => {
      expect(screen.getByText('Item 2 of 31')).toBeInTheDocument()
    })

    fireEvent.keyDown(document, { key: 'k' })
    await waitFor(() => {
      expect(screen.getByText('Item 1 of 31')).toBeInTheDocument()
    })
  })

  it('preserves per-card copy confirmation behavior', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')

    const copyContentButton = await screen.findByRole('button', { name: 'Copy Content' })
    fireEvent.click(copyContentButton)

    await waitFor(() => {
      expect(mocks.clipboardWriteText).toHaveBeenCalledWith('Content 1')
      expect(mocks.messageSuccess).toHaveBeenCalledWith('Content copied')
    })
  })

  it('hides transcript timings by default in cards and copy, with toggle to show', async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      const idMatch = path.match(/\/api\/v1\/media\/([^?]+)/)
      const id = idMatch ? Number(idMatch[1]) : null
      return {
        id: id ?? 1,
        title: `Item ${id ?? 1}`,
        type: 'audio',
        content: `00:12 Content ${id ?? 1}`
      }
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(screen.getByText('Content 1')).toBeInTheDocument()
      expect(screen.queryByText('00:12 Content 1')).not.toBeInTheDocument()
    })

    const copyContentButton = await screen.findByRole('button', { name: 'Copy Content' })
    fireEvent.click(copyContentButton)

    await waitFor(() => {
      expect(mocks.clipboardWriteText).toHaveBeenCalledWith('Content 1')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Show timings' }))

    await waitFor(() => {
      expect(screen.getByText('00:12 Content 1')).toBeInTheDocument()
    })

    fireEvent.click(copyContentButton)

    await waitFor(() => {
      expect(mocks.clipboardWriteText).toHaveBeenCalledWith('00:12 Content 1')
    })
  })

  it('preserves failed-item recovery behavior by allowing reselection retry', async () => {
    const attempts = new Map<number, number>()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      const idMatch = path.match(/\/api\/v1\/media\/([^?]+)/)
      const id = idMatch ? Number(idMatch[1]) : null
      const numericId = id ?? 1
      const attempt = (attempts.get(numericId) ?? 0) + 1
      attempts.set(numericId, attempt)
      if (numericId === 1 && attempt === 1) {
        throw new Error('detail fetch failed')
      }
      return {
        id: numericId,
        title: `Item ${numericId}`,
        type: 'pdf',
        content: `Recovered Content ${numericId}`
      }
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(attempts.get(1)).toBe(1)
      expect(screen.getByText('Select items on the left to view here.')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(screen.getByText('Recovered Content 1')).toBeInTheDocument()
    })
    expect(attempts.get(1)).toBeGreaterThanOrEqual(2)
  })

  it('prunes stale restored selections after 404 without retry loops', async () => {
    let staleFetchAttempts = 0
    mocks.getSetting
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce([150])
      .mockResolvedValueOnce(150)

    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      const idMatch = path.match(/\/api\/v1\/media\/([^?]+)/)
      const id = idMatch ? Number(idMatch[1]) : null
      const numericId = id ?? 1
      if (numericId === 150) {
        staleFetchAttempts += 1
        const notFoundError = new Error('Not Found') as Error & { status?: number }
        notFoundError.status = 404
        throw notFoundError
      }
      return {
        id: numericId,
        title: `Item ${numericId}`,
        type: 'pdf',
        content: `Content ${numericId}`
      }
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(staleFetchAttempts).toBeGreaterThan(0)
    })

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(staleFetchAttempts).toBe(1)
  })

  it('forces list mode and shows tab switcher on mobile viewports', async () => {
    setMobileViewport(true)
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('resizable-panels-collapsed')).toBeInTheDocument()
    })
    // Mobile shows tab bar with Filters, Results, Content
    expect(screen.getByTestId('mobile-tab-bar')).toBeInTheDocument()
    expect(screen.getByTestId('mobile-tab-0')).toBeInTheDocument()
    expect(screen.getByTestId('mobile-tab-1')).toBeInTheDocument()
    expect(screen.getByTestId('mobile-tab-2')).toBeInTheDocument()
    // Default tab is Results (index 1)
    // Switch to Content tab to see reading pane
    fireEvent.click(screen.getByTestId('mobile-tab-2'))
    await waitFor(() => {
      expect(screen.getByText('Focus')).toBeInTheDocument()
    })
  })

  it('keeps mobile viewer in single-item mode for multi-selection', async () => {
    setMobileViewport(true)
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByTestId('resizable-panels-collapsed')).toBeInTheDocument()
    })

    // Switch to Results tab to select items
    fireEvent.click(screen.getByTestId('mobile-tab-1'))

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    // Switch to Content tab to see viewer
    fireEvent.click(screen.getByTestId('mobile-tab-2'))

    await waitFor(() => {
      expect(screen.getByText('Single item view')).toBeInTheDocument()
    })
  })

  it('keeps list and viewer virtualization counts aligned with result/selection state', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    expect(
      mocks.useVirtualizer.mock.calls.some(
        (call) => Number((call[0] as { count?: number })?.count) === mediaItems.length
      )
    ).toBe(true)
    expect(
      mocks.useVirtualizer.mock.calls.some(
        (call) => Number((call[0] as { count?: number })?.count) === 0
      )
    ).toBe(true)

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByText('2 / 30 selected')).toBeInTheDocument()
    })
    expect(
      mocks.useVirtualizer.mock.calls.some(
        (call) => Number((call[0] as { count?: number })?.count) === 2
      )
    ).toBe(true)
  })

  it('preserves existing non-visible selection when Ctrl+A selects visible items', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 31')
    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })

    fireEvent.keyDown(document, { key: 'a', ctrlKey: true })

    await waitFor(() => {
      expect(screen.getByText('30 / 30 selected')).toBeInTheDocument()
    })
    expect(within(getResultRowByTitle('Item 31')).getByRole('checkbox')).toBeChecked()
  })

  it('shows explicit add/replace selection actions in the options menu', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /options/i }))

    expect(
      screen.getByRole('button', {
        name: /add visible to selection/i
      })
    ).toBeInTheDocument()
    expect(
      screen.getByRole('button', {
        name: /replace selection with visible/i
      })
    ).toBeInTheDocument()
  })

  it('shows cross-page selection scope text for current selection count', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByText(/selected across pages: 2/i)).toBeInTheDocument()
    })
  })

  it('opens selected-items drawer and allows removing a selected item', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByText('2 / 30 selected')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId('view-selected-items-button'))

    await waitFor(() => {
      expect(screen.getByRole('dialog', { name: /selected items/i })).toBeInTheDocument()
    })

    const drawer = screen.getByRole('dialog', { name: /selected items/i })
    fireEvent.click(within(drawer).getAllByRole('button', { name: 'Remove from selection' })[0])

    await waitFor(() => {
      expect(screen.getByText('1 / 30 selected')).toBeInTheDocument()
    })
  })

  it('shows explicit IA guidance linking sidebar, viewer, and open-items jump controls', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    expect(
      screen.getByText(/click to preview\. use checkboxes to select/i)
    ).toBeInTheDocument()

    selectItemByCheckbox('Item 1')

    await waitFor(() => {
      expect(
        screen.getByText(/search\/filter in sidebar, inspect in viewer, jump using open items/i)
      ).toBeInTheDocument()
    })
  })

  it('shows inline auto-mode transition banner when view mode auto-switches', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')
    selectItemByCheckbox('Item 3')
    selectItemByCheckbox('Item 4')
    selectItemByCheckbox('Item 5')

    await waitFor(() => {
      expect(screen.getByText(/auto-switched to stack view/i)).toBeInTheDocument()
    })
  })

  it('uses remove-from-selection label in compare cards instead of unstack', async () => {
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: 'Remove from selection' }).length).toBeGreaterThan(0)
    })
    expect(screen.queryByRole('button', { name: 'Unstack' })).not.toBeInTheDocument()
  })

  it('wraps long open-item labels instead of forcing one horizontal minimap line', async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request?.path || '')
      const idMatch = path.match(/\/api\/v1\/media\/([^?]+)/)
      const id = idMatch ? Number(idMatch[1]) : null
      return {
        id: id ?? 1,
        title: `Extremely long selected media title ${id ?? 1} that should wrap into another line in the open items strip`,
        type: 'document',
        content: `Content ${id ?? 1}`
      }
    })

    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    const openItemsRow = await screen.findByTestId('media-review-open-items')
    expect(openItemsRow).not.toHaveClass('overflow-x-auto')

    const firstOpenItemButton = screen.getByRole('button', {
      name: /1\.\s*Extremely long selected media title/i
    })
    expect(firstOpenItemButton.className).toContain('max-w-[42ch]')
    expect(firstOpenItemButton.className).toContain('whitespace-normal')
  })

  it('exposes named controls and keyboard focus progression for selection management actions', async () => {
    const user = userEvent.setup()
    render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')

    await waitFor(() => {
      expect(screen.getByText(/selected across pages: 2/i)).toBeInTheDocument()
    })

    const optionsButton = screen.getByRole('button', { name: /options/i })
    const viewSelectedItemsButton = screen.getByTestId('view-selected-items-button')
    const selectionStatus = screen.getByTestId('media-multi-selection-status')

    expect(optionsButton).toHaveAttribute('aria-haspopup', 'menu')
    expect(viewSelectedItemsButton).toBeInTheDocument()
    expect(selectionStatus).toHaveTextContent(/selection status:/i)

    optionsButton.focus()
    expect(optionsButton).toHaveFocus()
    await user.tab()
    expect(document.activeElement).not.toBe(optionsButton)
    expect((document.activeElement as HTMLElement).tagName.toLowerCase()).toBe('button')
  })

  it('has no axe violations for core aria naming and region rules', async () => {
    const { container } = render(<MediaReviewPage />)

    await waitFor(() => {
      expect(screen.getByText('0 / 30 selected')).toBeInTheDocument()
    })

    const results = await axe.run(container, {
      runOnly: {
        type: 'rule',
        values: [
          'aria-command-name',
          'aria-input-field-name',
          'aria-required-attr',
          'aria-valid-attr',
          'aria-valid-attr-value',
          'button-name',
          'focus-order-semantics',
          'region'
        ]
      }
    })

    expect(results.violations).toEqual([])
  })
})
