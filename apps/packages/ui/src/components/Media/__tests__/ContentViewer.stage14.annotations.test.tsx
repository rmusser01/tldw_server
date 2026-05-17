import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'
import { useMediaReadingProgress } from '@/hooks/useMediaReadingProgress'

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | {
            defaultValue?: string
            size?: string
            minutes?: number
            percent?: number
            timestamp?: string
          }
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
  id: 777,
  title: 'Annotation target',
  raw: {},
  meta: {
    type: 'document'
  }
}

describe('ContentViewer stage 14 annotations baseline', () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    vi.mocked(useMediaReadingProgress).mockReturnValue({
      saveProgress: vi.fn(),
      clearProgress: vi.fn(),
      progressPercent: null
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('supports create, update, sync, and delete actions in annotations panel', async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request?.path || '')
      if (path.endsWith('/outline')) {
        return { media_id: 777, has_outline: true, entries: [] }
      }
      if (path.endsWith('/insights')) {
        return { media_id: 777, insights: [] }
      }
      if (path.includes('/references')) {
        return { media_id: 777, references: [] }
      }
      if (path.includes('/figures')) {
        return { media_id: 777, figures: [] }
      }
      if (path.endsWith('/annotations') && request?.method === 'GET') {
        return { media_id: 777, annotations: [] }
      }
      if (path.endsWith('/annotations') && request?.method === 'POST') {
        return {
          id: 'ann-1',
          media_id: 777,
          location: 'manual:123',
          text: request?.body?.text || 'Created text',
          color: request?.body?.color || 'yellow',
          note: request?.body?.note,
          annotation_type: request?.body?.annotation_type || 'highlight',
          created_at: '2026-02-18T00:00:00.000Z',
          updated_at: '2026-02-18T00:00:00.000Z'
        }
      }
      if (path.endsWith('/annotations/ann-1') && request?.method === 'PUT') {
        return {
          id: 'ann-1',
          media_id: 777,
          location: 'manual:123',
          text: 'Manual annotation text',
          color: 'yellow',
          note: request?.body?.note || 'Updated note',
          annotation_type: 'highlight',
          created_at: '2026-02-18T00:00:00.000Z',
          updated_at: '2026-02-18T00:00:00.000Z'
        }
      }
      if (path.endsWith('/annotations/sync') && request?.method === 'POST') {
        return { media_id: 777, synced_count: 1, annotations: [] }
      }
      if (path.endsWith('/annotations/ann-1') && request?.method === 'DELETE') {
        return { success: true }
      }
      return {}
    })

    render(
      <ContentViewer
        selectedMedia={selectedMedia}
        content={'Body text'}
        mediaDetail={{ type: 'document' }}
      />
    )

    fireEvent.click(screen.getByTestId('media-intelligence-toggle'))
    fireEvent.click(await screen.findByTestId('media-intelligence-tab-annotations'))

    await waitFor(() => {
      expect(screen.getByTestId('media-annotation-manual-text')).toBeInTheDocument()
    })

    fireEvent.change(screen.getByTestId('media-annotation-manual-text'), {
      target: { value: 'Manual annotation text' }
    })
    fireEvent.change(screen.getByTestId('media-annotation-note-input'), {
      target: { value: 'Initial note' }
    })
    fireEvent.click(screen.getByTestId('media-annotation-create'))

    await waitFor(() => {
      expect(screen.getByTestId('media-intelligence-annotation-item')).toHaveTextContent(
        'Manual annotation text'
      )
    })

    const promptSpy = vi.spyOn(window, 'prompt').mockReturnValue('Updated note')
    fireEvent.click(screen.getByTestId('media-annotation-edit-ann-1'))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/api/v1/media/777/annotations/ann-1',
          method: 'PUT',
          body: { note: 'Updated note' }
        })
      )
    })
    promptSpy.mockRestore()

    fireEvent.click(screen.getByTestId('media-annotation-sync'))
    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/api/v1/media/777/annotations/sync',
          method: 'POST'
        })
      )
    })

    fireEvent.click(screen.getByTestId('media-annotation-delete-ann-1'))
    await waitFor(() => {
      expect(screen.queryByTestId('media-annotation-delete-ann-1')).not.toBeInTheDocument()
    })
  })

  it('captures selected content text into annotation draft and saves highlight', async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request?.path || '')
      if (path.endsWith('/outline')) return { media_id: 777, has_outline: true, entries: [] }
      if (path.endsWith('/insights')) return { media_id: 777, insights: [] }
      if (path.includes('/references')) return { media_id: 777, references: [] }
      if (path.includes('/figures')) return { media_id: 777, figures: [] }
      if (path.endsWith('/annotations') && request?.method === 'GET') {
        return { media_id: 777, annotations: [] }
      }
      if (path.endsWith('/annotations') && request?.method === 'POST') {
        return {
          id: 'ann-2',
          media_id: 777,
          location: request?.body?.location || 'selection:1',
          text: request?.body?.text || '',
          color: request?.body?.color || 'yellow',
          note: request?.body?.note,
          annotation_type: request?.body?.annotation_type || 'highlight',
          created_at: '2026-02-18T00:00:00.000Z',
          updated_at: '2026-02-18T00:00:00.000Z'
        }
      }
      return {}
    })

    render(
      <ContentViewer
        selectedMedia={selectedMedia}
        content={'Selected body text for annotation capture'}
        mediaDetail={{ type: 'document' }}
      />
    )

    const contentNode = screen.getByText('Selected body text for annotation capture')
    const textNode = contentNode.firstChild
    expect(textNode).not.toBeNull()

    const selection = window.getSelection()
    expect(selection).not.toBeNull()
    const range = document.createRange()
    range.setStart(textNode as Text, 0)
    range.setEnd(textNode as Text, 'Selected body text'.length)
    selection!.removeAllRanges()
    selection!.addRange(range)

    fireEvent.mouseUp(contentNode)

    await waitFor(() => {
      expect(screen.getByTestId('media-selection-actions-popover')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('media-annotation-selection-preview')).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId('media-selection-action-annotate'))

    await waitFor(() => {
      expect(screen.getByTestId('media-annotation-selection-preview')).toHaveTextContent(
        'Selected body text'
      )
    })

    fireEvent.click(screen.getByTestId('media-annotation-create'))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: '/api/v1/media/777/annotations',
          method: 'POST',
          body: expect.objectContaining({
            text: 'Selected body text',
            annotation_type: 'highlight'
          })
        })
      )
    })
  })

  it('clears selection actions when the selected media changes', async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request?.path || '')
      if (path.endsWith('/outline')) return { media_id: 777, has_outline: true, entries: [] }
      if (path.endsWith('/insights')) return { media_id: 777, insights: [] }
      if (path.includes('/references')) return { media_id: 777, references: [] }
      if (path.includes('/figures')) return { media_id: 777, figures: [] }
      if (path.endsWith('/annotations') && request?.method === 'GET') {
        return { media_id: 777, annotations: [] }
      }
      return {}
    })

    const { rerender } = render(
      <ContentViewer
        selectedMedia={selectedMedia}
        content={'Original selected body text'}
        mediaDetail={{ type: 'document' }}
      />
    )

    const contentNode = screen.getByText('Original selected body text')
    const textNode = contentNode.firstChild
    expect(textNode).not.toBeNull()

    const selection = window.getSelection()
    expect(selection).not.toBeNull()
    const range = document.createRange()
    range.setStart(textNode as Text, 0)
    range.setEnd(textNode as Text, 'Original selected'.length)
    selection!.removeAllRanges()
    selection!.addRange(range)

    fireEvent.mouseUp(contentNode)

    await waitFor(() => {
      expect(screen.getByTestId('media-selection-actions-popover')).toBeInTheDocument()
    })

    rerender(
      <ContentViewer
        selectedMedia={{
          ...selectedMedia,
          id: 778,
          title: 'Next annotation target'
        }}
        content={'Next media body text'}
        mediaDetail={{ type: 'document' }}
      />
    )

    await waitFor(() => {
      expect(screen.queryByTestId('media-selection-actions-popover')).not.toBeInTheDocument()
    })
    expect(screen.queryByTestId('media-annotation-selection-preview')).not.toBeInTheDocument()
  })
})
