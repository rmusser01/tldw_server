import { tldwClient } from '@/services/tldw/TldwApiClient'
import * as mediaHandoff from "@/services/tldw/media-chat-handoff"
import React from 'react'
import { act, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axe from 'axe-core'
import MediaReviewPage from '../MediaReviewPage'
import { useMediaReviewState } from '../hooks/useMediaReviewState'
import { useMediaReviewActions } from '../hooks/useMediaReviewActions'

const mediaItems = Array.from({ length: 40 }).map((_, idx) => ({
  id: idx + 1,
  title: `Item ${idx + 1}`,
  snippet: `Snippet ${idx + 1}`,
  type: 'pdf',
  created_at: '2026-02-17T00:00:00.000Z'
}))

vi.mock("@/utils/research-workspace-prefill", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/utils/research-workspace-prefill")>(),
  queueResearchWorkspacePrefill: (...args: unknown[]) => mocks.queuePrefill(...args)
}))

vi.mock('@/hooks/useHomeMilestoneScope', () => ({ useHomeMilestoneScope: () => 'server:alice' }))

const mocks = vi.hoisted(() => ({
  authorityKey: 'verified-alice' as string | null,
  authorityRevision: 0,
  locationKey: 'initial',
  realFocusDropdown: false,
  items: null as any,
  queuePrefill: vi.fn().mockResolvedValue(undefined),
  downloadBlob: vi.fn(),
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

vi.mock('@/services/tldw/quick-ingest-authority', () => ({
  useQuickIngestAuthority: () => mocks.authorityKey,
  quickIngestAuthority: { capture: () => {
    const owner = mocks.authorityKey
    if (!owner) throw new Error('Unverified owner')
    const revision = mocks.authorityRevision
    return { authorityKey: owner, authorityRevision: revision, requestScope: { config: { serverUrl: 'http://fixture.invalid', authMode: 'multi-user' }, userId: owner }, isCurrent: () => owner === mocks.authorityKey && revision === mocks.authorityRevision, signal: new AbortController().signal }
  } }
}))

vi.mock('@/utils/download-blob', () => ({ downloadBlob: mocks.downloadBlob }))
vi.mock('@/hooks/useMediaCapabilities', () => ({ useMediaCapabilities: () => ({canDelete: true, loading: false}) }))

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
  useNavigate: () => mocks.navigate,
  useLocation: () => ({ key: mocks.locationKey })
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
    data: mocks.items ?? mediaItems,
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
  MEDIA_REVIEW_SELECTION_SNAPSHOT_SETTING: { key: 'media-review-selection-snapshot', defaultValue: null },
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
    if (mocks.realFocusDropdown && rest.className === "min-w-[12rem]") return <div data-testid="real-focus-dropdown"><actual.Select value={value} onChange={onChange} options={options} {...rest} /></div>
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
              .flatMap((item: any) => item?.children ?? [item])
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

describe('MediaReviewPage active reading context', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    mocks.authorityKey = 'verified-alice'
    mocks.authorityRevision = 0
    mocks.locationKey = 'initial'
    mocks.realFocusDropdown = false
    mocks.items = null
    mocks.downloadBlob.mockReset()
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

  it('makes pointer and Enter preview the same item without selecting it, with named reachable checkbox', async () => {
    render(<MediaReviewPage />)
    const row = getResultRowByTitle('Item 2')
    fireEvent.click(row)
    await screen.findByText('Content 2')
    expect(screen.getByTestId('media-review-reading-context')).toHaveTextContent('Result preview: Item 2')
    const checkbox = within(row).getByRole('checkbox', { name: 'Select Item 2' })
    expect(checkbox.tabIndex).toBe(0)
    expect(checkbox).not.toBeChecked()
    fireEvent.keyDown(getResultRowByTitle('Item 3'), { key: 'Enter' })
    await screen.findByText('Content 3')
    expect(screen.getByTestId('media-review-reading-context')).toHaveTextContent('Result preview: Item 3')
    selectItemByCheckbox('Item 1')
    expect(screen.getByText('Content 3')).toBeInTheDocument()
    expect(mocks.bgRequest.mock.calls.filter(([r]) => String(r.path).startsWith('/api/v1/media/1?'))).toHaveLength(0)
  })

  it("hands all 40 cross-page selected sources to Ask and Research with retained titles and types", async () => {
    const selected = mediaItems.map(item => ({ ...item, type: item.id === 1 ? "audio" : "pdf" }))
    mocks.items = selected.slice(0, 20)
    mocks.queuePrefill.mockClear()
    mocks.navigate.mockClear()
    const { rerender } = render(<MediaReviewPage />)
    for (let id = 1; id <= 20; id++) selectItemByCheckbox(`Item ${id}`)
    mocks.items = selected.slice(20)
    rerender(<MediaReviewPage />)
    for (let id = 21; id <= 40; id++) selectItemByCheckbox(`Item ${id}`)
    const toolbar = screen.getByTestId("media-multi-batch-toolbar")
    fireEvent.click(within(toolbar).getByRole("button", { name: "Ask selected items" }))
    expect(new URL(mocks.navigate.mock.calls.at(-1)![0], "https://local.test").searchParams.get("media_ids")).toBe(selected.map(item => item.id).join(","))
    fireEvent.click(within(toolbar).getByRole("button", { name: "Research with selected sources" }))
    await waitFor(() => expect(mocks.queuePrefill).toHaveBeenCalledOnce())
    expect(mocks.queuePrefill.mock.calls[0][0].sources.map((source: any) => ({ mediaId: source.mediaId, title: source.title, type: source.type }))).toEqual(selected.map(item => ({ mediaId: item.id, title: item.title, type: item.type })))
    expect(mocks.bgRequest.mock.calls.filter(([r]) => /media\/\d+\?/.test(String(r.path)))).toHaveLength(0)
  })

  it('retains 40 selected metadata IDs but fetches only the active 30-item reading window', async () => {
    render(<MediaReviewPage />)
    for (let id = 1; id <= 40; id++) selectItemByCheckbox(`Item ${id}`)
    expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('40 selected')
    expect(mocks.bgRequest.mock.calls.filter(([r]) => /media\/\d+\?/.test(String(r.path)))).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: 'Review selected (40)' }))
    await waitFor(() => expect(mocks.bgRequest.mock.calls.filter(([r]) => /media\/\d+\?/.test(String(r.path)))).toHaveLength(30))
    expect(screen.getByTestId('media-review-reading-window')).toHaveTextContent('Reading 1–30 of 40 selected (30 at a time)')
    expect(screen.queryByText('Content 31')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Next reading window' }))
    await screen.findByText('Content 40')
    expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('40 selected')
    expect(screen.getByTestId('media-review-reading-window')).toHaveTextContent('Reading 31–40 of 40 selected')
    expect(screen.queryByText('Content 1')).not.toBeInTheDocument()
    expect(screen.getByText('Item 31 of 40')).toBeInTheDocument()
  })

  it('navigates ordered selection across result pages and restores retained preview', async () => {
    const { rerender } = render(<MediaReviewPage />)
    selectItemByCheckbox('Item 2')
    selectItemByCheckbox('Item 1')
    mocks.items = mediaItems.slice(30)
    rerender(<MediaReviewPage />)
    fireEvent.click(getResultRowByTitle('Item 40'))
    await screen.findByText('Content 40')
    fireEvent.click(screen.getByRole('button', { name: 'Review selected (2)' }))
    await screen.findByText('Content 2')
    expect(screen.getByText('Item 1 of 2')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByText('Item 2 of 2')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Return to preview' }))
    expect(screen.getByText('Content 40')).toBeInTheDocument()
  })

  it('keeps mobile checkbox assembly on Results, opens preview and selected review in Content with Back', async () => {
    setMobileViewport(true)
    render(<MediaReviewPage />)
    selectItemByCheckbox('Item 1')
    expect(screen.getByTestId('media-review-results-list')).toBeInTheDocument()
    fireEvent.click(getResultRowByTitle('Item 2'))
    await screen.findByText('Content 2')
    expect(screen.queryByTestId('media-review-results-list')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Back to results' }))
    expect(within(getResultRowByTitle('Item 1')).getByRole('checkbox')).toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: 'Review selected (1)' }))
    await screen.findByText('Content 1')
    expect(screen.getByText('Item 1 of 1')).toBeInTheDocument()
  })

  it('accepts only the atomic owned snapshot, refreshes same-route handoff, and rejects late stale content', async () => {
    let snapshot: any = {version: 1, authorityKey: 'verified-bob', selectedIds: [40]}
    mocks.getSetting.mockImplementation(async (setting) => setting.key === 'media-review-selection-snapshot' ? snapshot : setting.key === 'mediaReviewSelection' ? [39] : null)
    const { rerender } = render(<MediaReviewPage />)
    await waitFor(() => expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected'))
    expect(mocks.bgRequest.mock.calls.filter(([r]) => /media\/\d+\?/.test(String(r.path)))).toHaveLength(0)
    snapshot = {version: 1, authorityKey: 'verified-alice', selectedIds: [2, '2', 1]}
    mocks.locationKey = 'handoff'
    rerender(<MediaReviewPage />)
    await screen.findByText('Content 2')
    expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('2 selected')
    let resolve: any
    mocks.bgRequest.mockImplementation(() => new Promise(r => {resolve = r}))
    fireEvent.click(getResultRowByTitle('Item 3'))
    mocks.authorityKey = 'verified-bob'
    rerender(<MediaReviewPage />)
    resolve({id: 3, title:'Old owner', content:'Old owner secret'})
    await waitFor(() => expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected'))
    expect(screen.queryByText('Old owner secret')).not.toBeInTheDocument()
  })
  it('sends all 40 selected IDs to batch actions and exports missing content on demand', async () => {
    const tags = vi.spyOn(tldwClient, 'bulkUpdateMediaKeywords').mockResolvedValue({updated:40} as any)
    const reprocess = vi.spyOn(tldwClient, 'reprocessMedia').mockResolvedValue({} as any)
    const trash = vi.spyOn(tldwClient, 'deleteMedia').mockResolvedValue(undefined as any)
    render(<MediaReviewPage />)
    for (let id = 1; id <= 40; id++) selectItemByCheckbox(`Item ${id}`)
    fireEvent.change(screen.getByTestId('media-multi-batch-keywords'), {target:{value:'research'}})
    fireEvent.click(screen.getByTestId('media-multi-batch-add-tags'))
    await waitFor(() => expect(tags).toHaveBeenCalledWith({media_ids: mediaItems.map(item => item.id), keywords:['research'], mode:'add'}, expect.objectContaining({signal: expect.any(AbortSignal)})))
    fireEvent.click(screen.getByTestId('media-multi-batch-reprocess'))
    await waitFor(() => expect(reprocess).toHaveBeenCalledTimes(40))
    fireEvent.click(screen.getByTestId('media-multi-batch-export'))
    await waitFor(() => expect(mocks.downloadBlob).toHaveBeenCalledTimes(1))
    const blob = mocks.downloadBlob.mock.calls[0][0]
    const exported = await new Promise<string>((resolve) => {const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob)})
    expect(JSON.parse(exported).items).toHaveLength(40)
    expect(exported).toContain('Content 40')
    expect(screen.queryByText('Content 40')).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('media-multi-batch-trash'))
    await waitFor(() => expect(trash).toHaveBeenCalledTimes(40))
    await waitFor(() => expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('0 selected'))
  })

  it('bounds simultaneous requests while switching windows before prior detail loads finish', async () => {
    const pending: Array<() => void> = []
    let inFlight = 0
    let maximum = 0
    mocks.bgRequest.mockImplementation(({path}) => new Promise(resolve => {
      const id = Number(String(path).match(/media\/(\d+)\?/)?.[1])
      inFlight++
      maximum = Math.max(maximum, inFlight)
      pending.push(() => {inFlight--; resolve({id, title:`Item ${id}`, content:`Content ${id}`})})
    }))
    render(<MediaReviewPage />)
    for (let id = 1; id <= 40; id++) selectItemByCheckbox(`Item ${id}`)
    fireEvent.click(screen.getByRole('button', {name:'Review selected (40)'}))
    expect(inFlight).toBe(30)
    fireEvent.click(screen.getByRole('button', {name:'Next reading window'}))
    expect(inFlight).toBe(30)
    pending.splice(0).forEach(resolve => resolve())
    await waitFor(() => expect(inFlight).toBe(10))
    pending.splice(0).forEach(resolve => resolve())
    await screen.findByText('Content 40')
    expect(maximum).toBeLessThanOrEqual(30)
    expect(screen.queryByText('Content 1')).not.toBeInTheDocument()
  })

  it('shares the request ceiling with on-demand export during an unresolved reading window', async () => {
    let inFlight = 0
    const pending: Array<() => void> = []
    mocks.bgRequest.mockImplementation(({path}) => new Promise(resolve => {
      const id = Number(String(path).match(/media\/(\d+)\?/)?.[1])
      inFlight++
      pending.push(() => {inFlight--; resolve({id, title:`Item ${id}`, content:`Content ${id}`})})
    }))
    const {unmount} = render(<MediaReviewPage />)
    for (let id = 1; id <= 40; id++) selectItemByCheckbox(`Item ${id}`)
    fireEvent.click(screen.getByRole('button', {name:'Review selected (40)'}))
    expect(inFlight).toBe(30)
    fireEvent.click(screen.getByTestId('media-multi-batch-export'))
    expect(inFlight).toBeLessThanOrEqual(30)
    unmount()
  })

  it('preserves selection made while owned snapshot restoration is pending', async () => {
    let resolve: any
    mocks.getSetting.mockImplementation(() => new Promise(r => {resolve = r}))
    render(<MediaReviewPage />)
    selectItemByCheckbox('Item 3')
    resolve({version:1, authorityKey:'verified-alice', selectedIds:[1,2]})
    await waitFor(() => expect(mocks.setSetting).toHaveBeenCalled())
    expect(within(getResultRowByTitle('Item 3')).getByRole('checkbox')).toBeChecked()
    expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('1 selected')
  })

  it('retains the preview navigation context when the search page changes', async () => {
    const {rerender} = render(<MediaReviewPage />)
    fireEvent.click(getResultRowByTitle('Item 2'))
    await screen.findByText('Content 2')
    mocks.items = mediaItems.slice(30)
    rerender(<MediaReviewPage />)
    expect(screen.getByText('Item 2 of 40')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', {name:'Next'}))
    await screen.findByText('Content 3')
    expect(screen.getByText('Item 3 of 40')).toBeInTheDocument()
  })

  it('applies keyboard selection to selected reading rather than a retained preview', async () => {
    render(<MediaReviewPage />)
    selectItemByCheckbox('Item 1')
    selectItemByCheckbox('Item 2')
    fireEvent.click(getResultRowByTitle('Item 40'))
    await screen.findByText('Content 40')
    fireEvent.click(screen.getByRole('button', {name:'Review selected (2)'}))
    await screen.findByText('Content 1')
    fireEvent.keyDown(document, {key:'x'})
    expect(screen.getByTestId('media-review-selection-count')).toHaveTextContent('1 selected')
    expect(within(getResultRowByTitle('Item 40')).getByRole('checkbox')).not.toBeChecked()
  })

  it('keeps the focused selected item inside its reading window after removal', async () => {
    mocks.getSetting.mockImplementation(async setting => setting.key === 'media-review-selection-snapshot' ? {version:1, authorityKey:'verified-alice', selectedIds:mediaItems.map(item => item.id)} : null)
    render(<MediaReviewPage />)
    await screen.findByText('Content 30')
    fireEvent.click(screen.getByRole('button', {name:'Next reading window'}))
    await screen.findByText('Content 40')
    fireEvent.click(screen.getByTestId('view-selected-items-button'))
    fireEvent.click(within(screen.getByTestId('selected-item-31')).getByRole('button', {name:'Remove from selection'}))
    await waitFor(() => expect(screen.getByTestId('media-review-reading-window')).toHaveTextContent('Reading 1–30 of 39 selected'))
    expect(screen.getByText('Item 1 of 39')).toBeInTheDocument()
  })

  it('reads the article body without its stored metadata envelope in multi review', async () => {
    const content = '[METADATA]\n{"url":"https://example.com/","content_hash":"fixture"}\n[/METADATA]\n\nArticle body'
    mocks.bgRequest.mockResolvedValue({ media_id: 1, source: { title: 'Example Domain', type: 'html' }, content: { text: content } })
    render(<MediaReviewPage />)
    fireEvent.click(getResultRowByTitle('Item 1'))
    const body = await screen.findByTestId('media-review-content-body-1')
    await waitFor(() => expect(body).toHaveTextContent('Article body'))
    expect(body).not.toHaveTextContent('[METADATA]')
  })

  it('uses nested source identity for off-page reading and export from the real detail DTO', async () => {
    mocks.bgRequest.mockResolvedValue({
      media_id: 99,
      source: { title: 'Example Domain', type: 'html', url: 'https://example.com/' },
      processing: { analysis: 'Stored analysis' },
      content: { text: 'Article body', word_count: 2 },
      keywords: [],
      versions: [],
      has_original_file: false
    })
    const { result } = renderHook(() => {
      const state = useMediaReviewState(React.useRef(null))
      return { state, actions: useMediaReviewActions(state) }
    })
    await act(async () => { await result.current.actions.ensureDetail(99) })
    expect(result.current.state.details[99]).toMatchObject({
      id: 99, title: 'Example Domain', type: 'html'
    })
    act(() => result.current.state.setSelectedIds([99]))
    await act(async () => { await result.current.actions.handleBatchExport('json') })
    const blob = mocks.downloadBlob.mock.calls[0][0] as Blob
    const exported = JSON.parse(await new Promise<string>(resolve => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.readAsText(blob)
    }))
    expect(exported.items[0]).toMatchObject({ title: 'Example Domain', type: 'html', content: 'Article body' })
  })

  it('does not recapture a replacement owner for the second comparison detail', async () => {
    let resolveLeft!: (detail: unknown) => void
    mocks.bgRequest.mockImplementation(({path}) => String(path).includes('/media/1?')
      ? new Promise(resolve => { resolveLeft = resolve })
      : Promise.resolve({id:2, content:'Replacement owner content'}))
    const {result, rerender} = renderHook(() => {
      const state = useMediaReviewState(React.useRef(null))
      return {state, actions:useMediaReviewActions(state)}
    })
    act(() => result.current.state.setSelectedIds([1,2]))
    let comparison!: Promise<void>
    act(() => { comparison = result.current.actions.handleCompareContent() })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    mocks.authorityKey = 'verified-bob'
    rerender()
    await act(async () => {
      resolveLeft({id:1, content:'Old owner content'})
      await comparison
    })
    expect(mocks.bgRequest.mock.calls.filter(([request]) => String(request.path).includes('/media/2?'))).toHaveLength(0)
    expect(result.current.state.details).toEqual({})
    expect(result.current.state.compareDiffOpen).toBe(false)
  })

  it('keeps an unread snapshot intact when the real registry storage read fails', async () => {
    const registry = await vi.importActual<typeof import('@/services/settings/registry')>('@/services/settings/registry')
    const storage = registry.getStorageForSetting({key:'media-review-selection-snapshot', defaultValue:null})
    vi.spyOn(storage, 'get').mockRejectedValue(new Error('Storage unavailable'))
    mocks.getSetting.mockImplementation(registry.getSetting)
    render(<MediaReviewPage />)
    await waitFor(() => expect(mocks.messageError).toHaveBeenCalledWith('Could not load the saved selection. Reopen Review to try again.'))
    expect(mocks.setSetting).not.toHaveBeenCalled()
    // Default registry callers retain the historical fallback behavior.
    await expect(registry.getSetting({key:'other-setting', defaultValue:null})).resolves.toBeNull()
  })

  it('persists an empty owned set after successful absence at the real registry boundary', async () => {
    const registry = await vi.importActual<typeof import('@/services/settings/registry')>('@/services/settings/registry')
    const storage = registry.getStorageForSetting({key:'media-review-selection-snapshot', defaultValue:null})
    vi.spyOn(storage, 'get').mockResolvedValue(undefined)
    mocks.getSetting.mockImplementation(registry.getSetting)
    render(<MediaReviewPage />)
    await waitFor(() => expect(mocks.setSetting).toHaveBeenCalledWith(expect.objectContaining({key:'media-review-selection-snapshot'}), {version:1, authorityKey:'verified-alice', selectedIds:[]}))
    expect(mocks.messageError).not.toHaveBeenCalled()
  })

  it('binds the real focus dropdown to the displayed preview after selected reading', async () => {
    setMobileViewport(true)
    mocks.realFocusDropdown = true
    render(<MediaReviewPage />)
    selectItemByCheckbox('Item 1')
    fireEvent.click(screen.getByRole('button', {name:'Review selected (1)'}))
    await screen.findByText('Content 1')
    fireEvent.click(screen.getByRole('button', {name:'Back to results'}))
    fireEvent.click(getResultRowByTitle('Item 2'))
    await screen.findByText('Content 2')
    expect(screen.getByTestId('real-focus-dropdown')).toHaveTextContent('2. Item 2')
  })

  it('preserves captured preview neighbors through the real focus dropdown after a search-page change', async () => {
    setMobileViewport(true)
    mocks.realFocusDropdown = true
    const {rerender} = render(<MediaReviewPage />)
    fireEvent.click(getResultRowByTitle('Item 2'))
    await screen.findByText('Content 2')
    mocks.items = mediaItems.slice(30)
    rerender(<MediaReviewPage />)
    await userEvent.click(within(screen.getByTestId('real-focus-dropdown')).getByRole('combobox'))
    await userEvent.click(await screen.findByText('3. Item 3'))
    await screen.findByText('Content 3')
    expect(screen.getByText('Item 3 of 40')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', {name:'Next'}))
    await screen.findByText('Content 4')
    expect(screen.getByText('Item 4 of 40')).toBeInTheDocument()
  })

  it("keeps the footer aligned with preview, selected reading, the next window and restored preview", async () => {
    render(<MediaReviewPage />)
    fireEvent.click(getResultRowByTitle("Item 3"))
    await screen.findByText("Content 3")
    const footer = screen.getByTestId("media-review-status-bar")
    expect(footer).toHaveTextContent("Previewing 3 of 40")
    fireEvent.click(
      screen.getByRole("button", { name: "Add visible to selection (40)" })
    )
    fireEvent.click(screen.getByRole("button", { name: "Review selected (40)" }))
    await waitFor(() =>
      expect(
        screen.getByTestId("media-review-reading-context")
      ).toHaveTextContent("Selected reading: Item 1")
    )
    expect(footer).toHaveTextContent("Reading item 1 of 40")
    expect(footer).not.toHaveTextContent("Previewing")
    fireEvent.click(
      within(screen.getByTestId("media-review-reading-window")).getByRole(
        "button", { name: "Next reading window" }
      )
    )
    await waitFor(() =>
      expect(
        screen.getByTestId("media-review-reading-context")
      ).toHaveTextContent("Selected reading: Item 31")
    )
    expect(footer).toHaveTextContent("40 selected")
    expect(footer).toHaveTextContent("Reading item 31 of 40")
    fireEvent.click(
      within(screen.getByTestId("media-review-reading-context")).getByRole(
        "button", { name: "Return to preview" }
      )
    )
    expect(screen.getByTestId("media-review-reading-context")).toHaveTextContent(
      "Result preview: Item 3"
    )
    expect(footer).toHaveTextContent("Previewing 3 of 40")
  })

  const retainedActions = [
    [
      "tags",
      (a: ReturnType<typeof useMediaReviewActions>) => a.handleBatchAddTags()
    ],
    [
      "trash",
      (a: ReturnType<typeof useMediaReviewActions>) => a.handleBatchMoveToTrash()
    ],
    [
      "export",
      (a: ReturnType<typeof useMediaReviewActions>) => a.handleBatchExport()
    ],
    [
      "reprocess",
      (a: ReturnType<typeof useMediaReviewActions>) => a.handleBatchReprocess()
    ],
    [
      "compare",
      (a: ReturnType<typeof useMediaReviewActions>) => a.handleCompareContent()
    ],
    [
      "chat",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a.handleChatAboutSelection()
    ],
    [
      "ensure",
      (a: ReturnType<typeof useMediaReviewActions>) => a.ensureDetail(9)
    ],
    ["retry", (a: ReturnType<typeof useMediaReviewActions>) => a.retryFetch(7)],
    [
      "filter",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a.runContentFiltering(mediaItems)
    ],
    [
      "filter bypass",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a.runContentFiltering(mediaItems)
    ],
    [
      "cancel filter",
      (a: ReturnType<typeof useMediaReviewActions>) => a.cancelContentFiltering()
    ],
    [
      "cached compare",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a.resolveDetailForCompare(7)
    ],
    [
      "list",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a._fetchList().catch(() => null)
    ],
    [
      "clear",
      (a: ReturnType<typeof useMediaReviewActions>) => a.clearSelectionWithGuard()
    ],
    [
      "replace",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a.replaceSelectionWithVisible()
    ],
    [
      "toggle",
      (a: ReturnType<typeof useMediaReviewActions>) => a.toggleSelect(9)
    ],
    [
      "add visible",
      (a: ReturnType<typeof useMediaReviewActions>) => a.addVisibleToSelection()
    ],
    [
      "remove",
      (a: ReturnType<typeof useMediaReviewActions>) => a.removeFromSelection(7)
    ],
    [
      "preview",
      (a: ReturnType<typeof useMediaReviewActions>) => a.previewItem(9)
    ],
    [
      "start reading",
      (a: ReturnType<typeof useMediaReviewActions>) => a.startSelectedReview()
    ],
    [
      "return preview",
      (a: ReturnType<typeof useMediaReviewActions>) => a.returnToPreview()
    ],
    [
      "window",
      (a: ReturnType<typeof useMediaReviewActions>) => a.changeReadingWindow(1)
    ],
    [
      "relative",
      (a: ReturnType<typeof useMediaReviewActions>) => a.goRelative(1)
    ],
    [
      "expand content",
      (a: ReturnType<typeof useMediaReviewActions>) => a.expandAllContent()
    ],
    [
      "collapse content",
      (a: ReturnType<typeof useMediaReviewActions>) => a.collapseAllContent()
    ],
    [
      "expand analysis",
      (a: ReturnType<typeof useMediaReviewActions>) => a.expandAllAnalysis()
    ],
    [
      "collapse analysis",
      (a: ReturnType<typeof useMediaReviewActions>) => a.collapseAllAnalysis()
    ],
    [
      "scroll",
      (a: ReturnType<typeof useMediaReviewActions>) => a.scrollToCard(7)
    ],
    [
      "open trash",
      (a: ReturnType<typeof useMediaReviewActions>) =>
        a.openTrashFromBatch([7, 8])
    ]
  ] as const

  describe.each([
    "before commit",
    "after commit",
    "round trip",
    "round trip before commit"
  ] as const)("retained initiating-owner actions %s", (phase) => {
    it.each(retainedActions)(
      "rejects %s before any request, cache, progress, selection or navigation change",
      async (_label, invoke) => {
        const tags = vi
          .spyOn(tldwClient, "bulkUpdateMediaKeywords")
          .mockResolvedValue({ updated: 2 } as any)
        const trash = vi
          .spyOn(tldwClient, "deleteMedia")
          .mockResolvedValue({} as any)
        const reprocess = vi
          .spyOn(tldwClient, "reprocessMedia")
          .mockResolvedValue({} as any)
        const handoff = vi
          .spyOn(mediaHandoff, "createMediaChatHandoff")
          .mockResolvedValue("retained-token")
        const mutations = new Map<string, ReturnType<typeof vi.fn>>()
        const scroll = vi.fn()
        const { result, rerender } = renderHook(() => {
          const state = useMediaReviewState(React.useRef(null))
          const guardedState = {
            ...state,
            data: mediaItems,
            allResults: mediaItems,
            visibleIds: [7, 8],
            viewerItems: [{ id: 7 }],
            cardRefs: { current: { "7": { scrollIntoView: scroll } } }
          } as typeof state
          for (const key of Object.keys(state).filter((key) =>
            key.startsWith("set")
          )) {
            if (!mutations.has(key)) mutations.set(key, vi.fn())
            ;(guardedState as any)[key] = mutations.get(key)
          }
          return { state, actions: useMediaReviewActions(guardedState) }
        })
        await act(async () => {})
        act(() => {
          result.current.state.setSelectedIds([7, 8])
          result.current.state.setBatchKeywordsDraft("private")
          result.current.state.setPreviewedId(7)
          result.current.state.setPreviewNavigationIds([7, 8, 9])
          result.current.state.setDetails({
            7: { id: 7, content: "Cached A content" }
          })
          result.current.state.setQuery("private")
          result.current.state.setIncludeContent(_label !== "filter bypass")
        })
        const retained = result.current.actions
        mocks.authorityKey = "verified-bob"
        mocks.authorityRevision++
        if (phase === "after commit" || phase === "round trip") rerender()
        if (phase === "round trip" || phase === "round trip before commit") {
          mocks.authorityKey = "verified-alice"
          mocks.authorityRevision++
          if (phase === "round trip") rerender()
        }
        mutations.forEach((spy) => spy.mockClear())
        mocks.bgRequest.mockClear()
        mocks.setSetting.mockClear()
        let value: unknown
        await act(async () => {
          value = await invoke(retained)
        })
        expect(
          [...mutations.entries()]
            .filter(([, spy]) => spy.mock.calls.length)
            .map(([key]) => key)
        ).toEqual([])
        expect(mocks.bgRequest).not.toHaveBeenCalled()
        expect(tags).not.toHaveBeenCalled()
        expect(trash).not.toHaveBeenCalled()
        expect(reprocess).not.toHaveBeenCalled()
        expect(handoff).not.toHaveBeenCalled()
        expect(mocks.downloadBlob).not.toHaveBeenCalled()
        expect(mocks.navigate).not.toHaveBeenCalled()
        expect(scroll).not.toHaveBeenCalled()
        if (_label === "cached compare") expect(value).toBeNull()
      }
    )
  })

  it.each(["clear", "replace", "trash"] as const)(
    "retires the retained %s toast continuation through A to B to A",
    async (action) => {
      vi.spyOn(tldwClient, "deleteMedia").mockResolvedValue({} as any)
      const { result, rerender } = renderHook(() => {
        const state = useMediaReviewState(React.useRef(null))
        return {
          state,
          actions: useMediaReviewActions({ ...state, allResults: mediaItems })
        }
      })
      await act(async () => {})
      act(() => result.current.state.setSelectedIds([7, 8]))
      await act(async () => {
        if (action === "clear") result.current.actions.clearSelectionWithGuard()
        else if (action === "replace")
          result.current.actions.replaceSelectionWithVisible()
        else await result.current.actions.handleBatchMoveToTrash()
      })
      const toast = (
        action === "trash" ? mocks.messageSuccess : mocks.messageInfo
      ).mock.calls.at(-1)![0]
      const toastView = render(toast)
      mocks.authorityKey = "verified-bob"
      mocks.authorityRevision++
      rerender()
      mocks.authorityKey = "verified-alice"
      mocks.authorityRevision++
      rerender()
      act(() => result.current.state.setSelectedIds([9]))
      fireEvent.click(within(toastView.container).getByRole("button"))
      expect(result.current.state.selectedIds).toEqual([9])
      expect(mocks.navigate).not.toHaveBeenCalled()
    }
  )

  it("keeps same-owner actions usable on initial StrictMode mount and fences deferred viewer focus", async () => {
    const focus = vi.fn()
    const { result, rerender } = renderHook(
      () => {
        const state = useMediaReviewState(React.useRef(null))
        return {
          state,
          actions: useMediaReviewActions({
            ...state,
            viewerRef: { current: { focus } } as any
          })
        }
      },
      { wrapper: React.StrictMode }
    )
    await act(async () => {})
    act(() => result.current.state.setSelectedIds([7, 8]))
    act(() => result.current.actions.startSelectedReview())
    expect(result.current.state.readingActive).toBe(true)
    mocks.authorityKey = "verified-bob"
    mocks.authorityRevision++
    rerender()
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5))
    })
    expect(focus).not.toHaveBeenCalled()
  })

  it("does not publish filtering cache or progress after a verified owner changes during detail loading", async () => {
    let finish!: (detail: unknown) => void
    mocks.bgRequest.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const { result } = renderHook(() => {
      const state = useMediaReviewState(React.useRef(null))
      return { state, actions: useMediaReviewActions(state) }
    })
    await act(async () => {})
    act(() => {
      result.current.state.setQuery("private")
      result.current.state.setIncludeContent(true)
    })
    let pending!: Promise<unknown>
    act(() => {
      pending = result.current.actions.runContentFiltering(mediaItems.slice(6, 8))
    })
    expect(result.current.state.contentFilterProgress.completed).toBe(0)
    mocks.authorityKey = "verified-bob"
    mocks.authorityRevision++
    await act(async () => {
      finish({ id: 7, content: "private" })
      await pending
    })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(result.current.state.details).toEqual({})
    expect(result.current.state.contentFilterProgress.completed).toBe(0)
  })

  it("does not revive retained actions when the same owner remounts the review hook", async () => {
    const mount = () =>
      renderHook(() => {
        const state = useMediaReviewState(React.useRef(null))
        return { state, actions: useMediaReviewActions(state) }
      })
    const first = mount()
    await act(async () => {})
    act(() => first.result.current.state.setSelectedIds([7, 8]))
    const retained = first.result.current.actions
    first.unmount()
    const second = mount()
    await act(async () => {})
    mocks.bgRequest.mockClear()
    await act(async () => {
      await retained.handleBatchExport()
      retained.openTrashFromBatch([7, 8])
    })
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.downloadBlob).not.toHaveBeenCalled()
    expect(mocks.navigate).not.toHaveBeenCalled()
    expect(second.result.current.state.selectedIds).toEqual([])
  })

  it("keeps fresh actions usable after a batched A to B to A verification without reviving retained actions", async () => {
    const tags = vi
      .spyOn(tldwClient, "bulkUpdateMediaKeywords")
      .mockResolvedValue({ updated: 2 } as any)
    const { result, rerender } = renderHook(
      () => {
        const state = useMediaReviewState(React.useRef(null))
        return { state, actions: useMediaReviewActions(state) }
      },
      { wrapper: React.StrictMode }
    )
    await act(async () => {})
    act(() => {
      result.current.state.setSelectedIds([7, 8])
      result.current.state.setBatchKeywordsDraft("same-owner")
    })
    const retained = result.current.actions
    mocks.authorityKey = "verified-bob"
    mocks.authorityRevision++
    mocks.authorityKey = "verified-alice"
    mocks.authorityRevision++
    rerender()
    await act(async () => {
      await retained.handleBatchAddTags()
    })
    expect(tags).not.toHaveBeenCalled()
    await act(async () => {
      await result.current.actions.handleBatchAddTags()
    })
    expect(tags).toHaveBeenCalledExactlyOnceWith(
      { media_ids: [7, 8], keywords: ["same-owner"], mode: "add" },
      expect.objectContaining({
        requestScope: expect.objectContaining({ userId: "verified-alice" })
      })
    )
  })


})
