import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'
import { useMediaReadingProgress } from '@/hooks/useMediaReadingProgress'
import { downloadBlob } from '@/utils/download-blob'

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  messageWarning: vi.fn()
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; size?: string; minutes?: number; percent?: number }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return fallbackOrOptions.defaultValue
          .replace('{{size}}', String(fallbackOrOptions.size ?? ''))
          .replace('{{minutes}}', String(fallbackOrOptions.minutes ?? ''))
          .replace('{{percent}}', String(fallbackOrOptions.percent ?? ''))
      }
      return key
    }
  })
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock('@/utils/download-blob', () => ({
  downloadBlob: vi.fn()
}))

vi.mock('antd', async (importOriginal) => {
  const actual = await importOriginal<typeof import('antd')>()
  const renderItems = (items: any[] | undefined): React.ReactElement[] => {
    if (!Array.isArray(items)) return []
    return items.flatMap((item) => {
      if (!item) return []
      if (item.type === 'divider') return []
      if (item.type === 'group' && Array.isArray(item.children)) {
        return renderItems(item.children)
      }
      return [
        <button
          key={String(item.key)}
          type="button"
          data-testid={`menu-item-${String(item.key)}`}
          onClick={() => item.onClick?.()}
        >
          {typeof item.label === 'string' ? item.label : String(item.key)}
        </button>
      ]
    })
  }

  return {
    ...actual,
    Select: () => <div />,
    Dropdown: ({ children, menu }: any) => (
      <div>
        {children}
        <div data-testid="mock-dropdown-menu">{renderItems(menu?.items)}</div>
      </div>
    ),
    Tooltip: ({ children }: any) => <>{children}</>,
    Modal: ({ open, children }: any) =>
      open ? <div data-testid="mock-modal">{children}</div> : null,
    Spin: () => null,
    message: {
      ...actual.message,
      success: mocks.messageSuccess,
      error: mocks.messageError,
      warning: mocks.messageWarning
    }
  }
})

vi.mock('@/hooks/useSetting', async () => {
  const React = await import('react')
  return {
    useSetting: (setting: { defaultValue: unknown }) => {
      const [value, setValue] = React.useState(setting.defaultValue)
      const setAsync = async (next: unknown | ((prev: unknown) => unknown)) => {
        setValue((prev) =>
          typeof next === 'function' ? (next as (prev: unknown) => unknown)(prev) : next
        )
      }
      return [value, setAsync, { isLoading: false }] as const
    }
  }
})

vi.mock('@/hooks/useMediaReadingProgress', () => ({
  useMediaReadingProgress: vi.fn()
}))

vi.mock('../AnalysisModal', () => ({ AnalysisModal: () => null }))
vi.mock('../AnalysisEditModal', () => ({ AnalysisEditModal: () => null }))
vi.mock('../VersionHistoryPanel', () => ({ VersionHistoryPanel: () => null }))
vi.mock('../DeveloperToolsSection', () => ({ DeveloperToolsSection: () => null }))
vi.mock('../DiffViewModal', () => ({ DiffViewModal: () => null }))
vi.mock('@/components/Common/MarkdownPreview', () => ({
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>
}))

const selectedMedia = {
  kind: 'media' as const,
  id: 42,
  title: 'Export target',
  raw: {},
  meta: {
    type: 'document',
    source: 'https://example.com/paper'
  }
}

describe('ContentViewer stage 14 export action', () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.messageWarning.mockReset()
    vi.mocked(downloadBlob).mockReset()
    vi.mocked(useMediaReadingProgress).mockReturnValue({
      saveProgress: vi.fn(),
      clearProgress: vi.fn(),
      progressPercent: null
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('opens export modal and exports markdown output', async () => {
    render(
      <ContentViewer
        selectedMedia={selectedMedia}
        content={'Body content for export'}
        mediaDetail={{ type: 'document', analysis: 'Analysis body' }}
      />
    )

    fireEvent.click(screen.getByTestId('menu-item-export-media'))
    expect(await screen.findByTestId('media-export-modal')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('media-export-format-markdown'))
    fireEvent.click(screen.getByTestId('media-export-confirm'))

    await waitFor(() => {
      expect(downloadBlob).toHaveBeenCalledTimes(1)
    })
    const [blob, filename] = vi.mocked(downloadBlob).mock.calls[0]
    expect(filename).toBe('export-target-42.md')
    expect(blob).toBeInstanceOf(Blob)
    await expect((blob as Blob).text()).resolves.toContain('# Export target')
    expect(mocks.messageSuccess).toHaveBeenCalledWith('Export ready.')
  })

  it('reads an extracted web body while exporting the complete stored source', async () => {
    const content = '[METADATA]\n{"url":"https://example.com/","content_hash":"fixture"}\n[/METADATA]\n\nArticle body'
    render(<ContentViewer selectedMedia={selectedMedia} content={content} contentDisplayMode="plain" />)
    expect(await screen.findByText('Article body')).toBeVisible()
    expect(screen.queryByText(/content_hash/)).toBeNull()
    fireEvent.click(screen.getByTestId('menu-item-export-media'))
    fireEvent.click(await screen.findByTestId('media-export-confirm'))
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledOnce())
    const parsed = JSON.parse(await (vi.mocked(downloadBlob).mock.calls[0][0] as Blob).text())
    expect(parsed.content).toBe(content)
  })

  it('exports json by default with structured payload', async () => {
    render(
      <ContentViewer
        selectedMedia={selectedMedia}
        content={'JSON body content'}
        mediaDetail={{ type: 'document', analysis: 'JSON analysis' }}
      />
    )

    fireEvent.click(screen.getByTestId('menu-item-export-media'))
    await screen.findByTestId('media-export-confirm')
    fireEvent.click(screen.getByTestId('media-export-confirm'))

    await waitFor(() => {
      expect(downloadBlob).toHaveBeenCalledTimes(1)
    })

    const [blob, filename] = vi.mocked(downloadBlob).mock.calls[0]
    expect(filename).toBe('export-target-42.json')
    const parsed = JSON.parse(await (blob as Blob).text())
    expect(parsed.id).toBe(42)
    expect(parsed.title).toBe('Export target')
    expect(parsed.content).toBe('JSON body content')
  })

  it('exports bibtex using safe metadata fields when selected', async () => {
    render(
      <ContentViewer
        selectedMedia={selectedMedia}
        content={'Body content for citation'}
        mediaDetail={{
          type: 'document',
          analysis: 'Citation analysis',
          safe_metadata: {
            doi: '10.1000/xyz123',
            journal: 'Journal of Testing',
            authors: ['Ada Lovelace', 'Grace Hopper'],
            year: 2024
          }
        }}
      />
    )

    fireEvent.click(screen.getByTestId('menu-item-export-media'))
    await screen.findByTestId('media-export-format-bibtex')
    fireEvent.click(screen.getByTestId('media-export-format-bibtex'))
    fireEvent.click(screen.getByTestId('media-export-confirm'))

    await waitFor(() => {
      expect(downloadBlob).toHaveBeenCalledTimes(1)
    })

    const [blob, filename] = vi.mocked(downloadBlob).mock.calls[0]
    expect(filename).toBe('export-target-42.bib')
    const text = await (blob as Blob).text()
    expect(text).toContain('@article{')
    expect(text).toContain('doi = {10.1000/xyz123}')
    expect(text).toContain('journal = {Journal of Testing}')
    expect(text).toContain('author = {Ada Lovelace and Grace Hopper}')
    expect(text).toContain('year = {2024}')
  })
})
