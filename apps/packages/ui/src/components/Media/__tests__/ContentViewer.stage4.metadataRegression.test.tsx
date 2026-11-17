import React from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'
import { useMediaReadingProgress } from '@/hooks/useMediaReadingProgress'

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
        | { defaultValue?: string; size?: string; minutes?: number; percent?: number; timestamp?: string }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return fallbackOrOptions.defaultValue
          .replace('{{size}}', String(fallbackOrOptions.size ?? ''))
          .replace('{{minutes}}', String(fallbackOrOptions.minutes ?? ''))
          .replace('{{percent}}', String(fallbackOrOptions.percent ?? ''))
          .replace('{{timestamp}}', String(fallbackOrOptions.timestamp ?? ''))
      }
      return key
    }
  })
}))

vi.mock('@/services/background-proxy', () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock('antd', async (importOriginal) => {
  const actual = await importOriginal<typeof import('antd')>()
  return {
    ...actual,
    Select: ({ onChange, value, ...rest }: any) => {
      const testId = typeof rest['data-testid'] === 'string' ? rest['data-testid'] : undefined
      if (!testId) {
        return <div />
      }
      return (
        <div data-testid={testId}>
          <button
            type="button"
            data-testid={`${testId}-set-first`}
            onClick={() => onChange?.(['first-keyword'])}
          >
            set first
          </button>
          <button
            type="button"
            data-testid={`${testId}-set-second`}
            onClick={() => onChange?.(['second-keyword'])}
          >
            set second
          </button>
          <span data-testid={`${testId}-value`}>
            {Array.isArray(value) ? value.join(',') : ''}
          </span>
        </div>
      )
    },
    Dropdown: ({ children }: any) => <>{children}</>,
    Tooltip: ({ children }: any) => <>{children}</>,
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

const baseSelectedMedia = {
  kind: 'media' as const,
  id: 777,
  title: 'Regression metadata item',
  raw: {},
  meta: {
    type: 'audio',
    source: 'https://example.org/source',
    duration: 360
  }
}

describe('ContentViewer stage 4 metadata regressions', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mocks.bgRequest.mockReset()
    mocks.bgRequest.mockImplementation(async (request: { body?: { keywords?: string[] } }) => ({ media_id: 777, keywords: request.body?.keywords || [] }))
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.messageWarning.mockReset()
    vi.mocked(useMediaReadingProgress).mockReturnValue({
      saveProgress: vi.fn(),
      clearProgress: vi.fn(),
      progressPercent: null
    })
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it('keeps title/type/source metadata visible in the header region', () => {
    render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'Body text'}
        mediaDetail={{ type: 'audio' }}
      />
    )

    expect(screen.getByText('Regression metadata item')).toBeInTheDocument()
    expect(screen.getByText('audio')).toBeInTheDocument()
    expect(screen.getByText('https://example.org/source')).toBeInTheDocument()
  })

  it('shows duration for media with duration and hides it when absent', () => {
    const { rerender } = render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'Body text'}
        mediaDetail={{ type: 'audio' }}
      />
    )

    expect(screen.getByText('06:00')).toBeInTheDocument()

    rerender(
      <ContentViewer
        selectedMedia={{
          ...baseSelectedMedia,
          meta: {
            ...baseSelectedMedia.meta,
            duration: undefined
          }
        }}
        content={'Body text'}
        mediaDetail={{ type: 'audio' }}
      />
    )

    expect(screen.queryByText('06:00')).not.toBeInTheDocument()
  })

  it('shows a reading progress chip when progress is available', () => {
    vi.mocked(useMediaReadingProgress).mockReturnValue({
      saveProgress: vi.fn(),
      clearProgress: vi.fn(),
      progressPercent: 41.7
    })

    render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'Body text'}
        mediaDetail={{ type: 'audio' }}
      />
    )

    expect(screen.getByTestId('media-reading-progress')).toHaveTextContent('42% read')
  })

  it('debounces keyword save requests and sends only the latest keyword set', async () => {
    render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'Body text'}
        mediaDetail={{ type: 'audio' }}
      />
    )

    fireEvent.click(screen.getByTestId('media-keywords-select-set-first'))
    fireEvent.click(screen.getByTestId('media-keywords-select-set-second'))

    vi.advanceTimersByTime(499)
    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/v1/media/777/keywords', method: 'PATCH' })
    )

    vi.advanceTimersByTime(1)
    await vi.runOnlyPendingTimersAsync()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/v1/media/777/keywords',
        method: 'PATCH',
        body: { keywords: ['second-keyword'], mode: 'set' }
      })
    )
  })
})


describe('persisted Media and Note keyword contracts', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mocks.bgRequest.mockReset().mockResolvedValue({})
    mocks.messageSuccess.mockReset()
    mocks.messageError.mockReset()
    mocks.messageWarning.mockReset()
    vi.mocked(useMediaReadingProgress).mockReturnValue({ saveProgress: vi.fn(), clearProgress: vi.fn(), progressPercent: null })
  })
  afterEach(() => { cleanup(); vi.runOnlyPendingTimers(); vi.useRealTimers() })

  it('persists the requested Media keywords rather than acknowledging a generic PUT no-op', async () => {
    let stored: string[] = []
    mocks.bgRequest.mockImplementation(async (request: { path: string; method: string; body?: { keywords?: string[]; mode?: string } }) => {
      if (request.path === '/api/v1/media/777/keywords' && request.method === 'PATCH' && request.body?.mode === 'set') stored = request.body?.keywords || []
      return { media_id: 777, keywords: stored }
    })
    render(<ContentViewer selectedMedia={baseSelectedMedia} content="Body text" mediaDetail={{ type: 'audio' }} />)
    fireEvent.click(screen.getByTestId('media-keywords-select-set-second'))
    await vi.advanceTimersByTimeAsync(500)
    expect(stored).toEqual(['second-keyword'])
  })

  it('keeps Note keywords on the versioned Note PUT contract', async () => {
    render(<ContentViewer selectedMedia={{ ...baseSelectedMedia, kind: 'note', raw: { version: 4 } }} content="Body text" mediaDetail={{ version: 4 }} />)
    fireEvent.click(screen.getByTestId('media-keywords-select-set-second'))
    await vi.advanceTimersByTimeAsync(500)
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/v1/notes/777', method: 'PUT', headers: expect.objectContaining({ 'expected-version': '4' }), body: { keywords: ['second-keyword'] } }))
  })
})
