import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: string | { defaultValue?: string; size?: string }) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return fallbackOrOptions.defaultValue.replace('{{size}}', String(fallbackOrOptions.size ?? ''))
      }
      return _key
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

vi.mock('../AnalysisModal', () => ({
  AnalysisModal: () => null
}))

vi.mock('../AnalysisEditModal', () => ({
  AnalysisEditModal: () => null
}))

vi.mock('../VersionHistoryPanel', () => ({
  VersionHistoryPanel: () => null
}))

vi.mock('../DeveloperToolsSection', () => ({
  DeveloperToolsSection: () => null
}))

vi.mock('../DiffViewModal', () => ({
  DiffViewModal: () => null
}))

vi.mock('@/components/Common/MarkdownPreview', () => ({
  MarkdownPreview: ({ content, size }: { content: string; size: string }) => (
    <div data-testid="markdown-preview" data-size={size}>
      {content}
    </div>
  )
}))

const selectedNote = {
  kind: 'note' as const,
  id: 101,
  title: 'Test note',
  raw: {},
  meta: { type: 'note' }
}

describe('ContentViewer stage 2 ergonomics', () => {
  it('applies text size controls to plain content rendering', async () => {
    render(
      <ContentViewer
        selectedMedia={selectedNote}
        content={'First line\nSecond line'}
        mediaDetail={{}}
        contentDisplayMode="plain"
      />
    )

    const contentRegion = screen.getByRole('region', { name: 'Media content' })
    const plainContent = contentRegion.querySelector('pre')
    expect(plainContent).toHaveClass('text-sm')

    fireEvent.click(screen.getByRole('button', { name: 'Text size L' }))

    expect(contentRegion.querySelector('pre')).toHaveClass('text-base')
  })

  it('shows and executes the back-to-top action after deep scroll', () => {
    const { container } = render(
      <ContentViewer selectedMedia={selectedNote} content={'Long content'} mediaDetail={{}} />
    )

    const scrollContainer = container.querySelector('div.overflow-y-auto') as HTMLDivElement
    expect(scrollContainer).toBeTruthy()

    const scrollToMock = vi.fn(({ top }: { top: number }) => {
      scrollContainer.scrollTop = top
    })
    ;(scrollContainer as any).scrollTo = scrollToMock

    scrollContainer.scrollTop = 700
    fireEvent.scroll(scrollContainer)

    const button = screen.getByRole('button', { name: 'Back to top' })
    expect(button).toBeInTheDocument()

    fireEvent.click(button)

    expect(scrollToMock).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' })
    expect(screen.queryByRole('button', { name: 'Back to top' })).not.toBeInTheDocument()
  })

  it('keeps markdown mode rendering active while text size changes', async () => {
    render(
      <ContentViewer
        selectedMedia={selectedNote}
        content={'# Heading\\n\\nBody copy'}
        mediaDetail={{}}
        contentDisplayMode="markdown"
      />
    )

    expect(await screen.findByTestId('markdown-preview')).toHaveAttribute('data-size', 'sm')

    fireEvent.click(screen.getByRole('button', { name: 'Text size S' }))

    expect(await screen.findByTestId('markdown-preview')).toHaveAttribute('data-size', 'xs')
  })
})
