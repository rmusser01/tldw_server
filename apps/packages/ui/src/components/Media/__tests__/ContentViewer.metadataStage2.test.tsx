import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; size?: string; minutes?: number; timestamp?: string }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return fallbackOrOptions.defaultValue
          .replace('{{size}}', String(fallbackOrOptions.size ?? ''))
          .replace('{{minutes}}', String(fallbackOrOptions.minutes ?? ''))
          .replace('{{timestamp}}', String(fallbackOrOptions.timestamp ?? ''))
      }
      return key
    }
  })
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

const baseSelectedMedia = {
  kind: 'media' as const,
  id: 909,
  title: 'Metadata heavy media',
  raw: {},
  meta: {
    type: 'document',
    source: 'scholar.example'
  }
}

describe('ContentViewer metadata stage 2 details', () => {
  it.each([
    [0, "not_indexed", "Not Indexed"],
    ["0", "not_indexed", "Not Indexed"],
    ["queued", "pending", "Pending"],
    [1, "completed", "Completed"],
    [-1, "failed", "Failed"],
  ])(
    "reports vector status %s without inventing a queued job",
    (value, status, label) => {
      render(
        <ContentViewer
          selectedMedia={baseSelectedMedia}
          content={"saved source"}
          mediaDetail={{
            processing: {
              chunking_status: "skipped",
              vector_processing_status: value,
            },
          }}
        />,
      );
      expect(
        screen.getByTestId("media-processing-status-vector"),
      ).toHaveAttribute("data-status", status);
      expect(
        screen.getByTestId("media-processing-status-vector"),
      ).toHaveTextContent(`Vector: ${label}`);
      fireEvent.click(screen.getByTestId("metadata-details-toggle"));
      expect(screen.getByTestId("metadata-processing-vector")).toHaveAttribute(
        "data-status",
        status,
      );
    },
  );

  it('renders processing badges and prioritized safe metadata fields', () => {
    render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'alpha beta gamma'}
        mediaDetail={{
          type: 'document',
          chunking_status: 'completed',
          vector_processing: 0,
          safe_metadata: {
            license: 'CC-BY-4.0',
            journal: 'Nature',
            doi: '10.1000/182',
            pmid: '1234567',
            extra_field: 'internal-id-7'
          }
        }}
      />
    )

    expect(screen.getByTestId('media-processing-status-chunking')).toHaveAttribute(
      'data-status',
      'completed'
    )
    expect(screen.getByTestId('media-processing-status-vector')).toHaveAttribute(
      'data-status',
      'not_indexed'
    )

    fireEvent.click(screen.getByTestId('metadata-details-toggle'))
    const panel = screen.getByTestId('metadata-details-panel')
    expect(panel).toBeInTheDocument()

    const rowIds = Array.from(
      panel.querySelectorAll<HTMLElement>('[data-testid^="metadata-field-"]')
    ).map((node) => node.dataset.testid)

    expect(rowIds.slice(0, 4)).toEqual([
      'metadata-field-doi',
      'metadata-field-pmid',
      'metadata-field-journal',
      'metadata-field-license'
    ])
    expect(screen.getByTestId('metadata-field-extra_field')).toBeInTheDocument()
  })

  it('updates processing badge states when API payload status changes', () => {
    const { rerender } = render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'status content'}
        mediaDetail={{ type: 'document', chunking_status: 'queued', vector_processing: 0 }}
      />
    )

    expect(screen.getByTestId('media-processing-status-chunking')).toHaveAttribute(
      'data-status',
      'pending'
    )
    expect(screen.getByTestId('media-processing-status-vector')).toHaveAttribute(
      'data-status',
      'not_indexed'
    )

    rerender(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'status content'}
        mediaDetail={{ type: 'document', chunking_status: 'completed', vector_processing: 1 }}
      />
    )

    expect(screen.getByTestId('media-processing-status-chunking')).toHaveAttribute(
      'data-status',
      'completed'
    )
    expect(screen.getByTestId('media-processing-status-vector')).toHaveAttribute(
      'data-status',
      'completed'
    )
  })

  it('shows empty-state copy while keeping keyword controls available', () => {
    render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'no metadata here'}
        mediaDetail={{ type: 'document' }}
      />
    )

    fireEvent.click(screen.getByTestId('metadata-details-toggle'))

    expect(screen.getByTestId('metadata-processing-empty')).toBeInTheDocument()
    expect(screen.getByTestId('metadata-safe-empty')).toBeInTheDocument()
    expect(screen.getByTestId('media-keywords-select')).toBeInTheDocument()
  })

  it('loads the collapsed metadata section body on demand', async () => {
    render(
      <ContentViewer
        selectedMedia={baseSelectedMedia}
        content={'metadata section content'}
        mediaDetail={{ type: 'document' }}
      />
    )

    fireEvent.click(screen.getByTitle('Metadata'))

    const panel = await screen.findByTestId('media-metadata-section-panel')
    expect(panel).toHaveTextContent('ID')
    expect(panel).toHaveTextContent('909')
    expect(panel).toHaveTextContent('Source')
    expect(panel).toHaveTextContent('scholar.example')
  })
})
