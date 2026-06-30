import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ContentViewer } from '../ContentViewer'

const registryLabels = vi.hoisted(() => ({
  loading: 'Loading via registry'
}))

const mocks = vi.hoisted(() => ({
  readAlongState: {
    status: 'idle',
    scope: null,
    activeSegmentId: null,
    activeIndex: -1,
    totalSegments: 0,
    error: null,
    cacheDisabled: false
  },
  startReadAlong: vi.fn(),
  pauseReadAlong: vi.fn(),
  resumeReadAlong: vi.fn(),
  stopReadAlong: vi.fn(),
  retryReadAlong: vi.fn(),
  skipReadAlong: vi.fn()
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || _key
    }
  })
}))

vi.mock('@/design-system', async (importActual) => {
  const actual = await importActual<typeof import('@/design-system')>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === 'loading' ? registryLabels.loading : state.label
        }
      }
    )
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

vi.mock('../read-along/useMediaReadAlongSession', () => ({
  useMediaReadAlongSession: () => ({
    state: mocks.readAlongState,
    activeSegmentId: mocks.readAlongState.activeSegmentId,
    start: mocks.startReadAlong,
    pause: mocks.pauseReadAlong,
    resume: mocks.resumeReadAlong,
    stop: mocks.stopReadAlong,
    retry: mocks.retryReadAlong,
    skip: mocks.skipReadAlong
  })
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
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>
}))

const mediaOne = {
  kind: 'media' as const,
  id: 1101,
  title: 'Accessibility Item One',
  raw: {},
  meta: { type: 'document' }
}

const mediaTwo = {
  kind: 'media' as const,
  id: 1102,
  title: 'Accessibility Item Two',
  raw: {},
  meta: { type: 'document' }
}

const makeRect = ({
  left,
  top,
  width,
  height
}: {
  left: number
  top: number
  width: number
  height: number
}): DOMRect =>
  ({
    x: left,
    y: top,
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    toJSON: () => ({})
  }) as DOMRect

const selectText = (node: HTMLElement, selectedText: string) => {
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT)
  let textNode: Node | null = node.firstChild
  let startOffset = 0
  while (textNode) {
    const value = textNode.textContent || ''
    const matchIndex = value.indexOf(selectedText)
    if (matchIndex >= 0) {
      startOffset = matchIndex
      break
    }
    textNode = walker.nextNode()
  }

  expect(textNode).not.toBeNull()
  const range = document.createRange()
  range.setStart(textNode as Text, startOffset)
  range.setEnd(textNode as Text, startOffset + selectedText.length)
  const selection = window.getSelection()
  expect(selection).not.toBeNull()
  selection!.removeAllRanges()
  selection!.addRange(range)
}

describe('ContentViewer stage 15 content announcements', () => {
  beforeEach(() => {
    window.getSelection()?.removeAllRanges()
    mocks.readAlongState = {
      status: 'idle',
      scope: null,
      activeSegmentId: null,
      activeIndex: -1,
      totalSegments: 0,
      error: null,
      cacheDisabled: false
    }
    mocks.startReadAlong.mockReset()
    mocks.pauseReadAlong.mockReset()
    mocks.resumeReadAlong.mockReset()
    mocks.stopReadAlong.mockReset()
    mocks.retryReadAlong.mockReset()
    mocks.skipReadAlong.mockReset()
  })

  it('announces loading and ready status when content state changes', async () => {
    const { rerender } = render(
      <ContentViewer
        selectedMedia={mediaOne}
        content=""
        mediaDetail={{ type: 'document' }}
        isDetailLoading
      />
    )

    const liveRegion = screen.getByTestId('content-selection-live-region')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Loading via registry Accessibility Item One')
    )

    rerender(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Body"
        mediaDetail={{ type: 'document' }}
        isDetailLoading={false}
      />
    )
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )
  })

  it('updates announcements for pointer, keyboard, and programmatic selection changes', async () => {
    const user = userEvent.setup()

    function Harness() {
      const [selected, setSelected] = React.useState(mediaOne)
      return (
        <div>
          <button type="button" onClick={() => setSelected(mediaOne)}>
            Select One
          </button>
          <button type="button" onClick={() => setSelected(mediaTwo)}>
            Select Two
          </button>
          <ContentViewer
            selectedMedia={selected}
            content="Body"
            mediaDetail={{ type: 'document' }}
          />
        </div>
      )
    }

    render(<Harness />)

    const liveRegion = screen.getByTestId('content-selection-live-region')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )

    await user.click(screen.getByRole('button', { name: 'Select Two' }))
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item Two')
    )

    const selectOneButton = screen.getByRole('button', { name: 'Select One' })
    selectOneButton.focus()
    await user.keyboard('{Enter}')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )
  })

  it('does not re-announce on unrelated rerenders for the same item state', async () => {
    const { rerender } = render(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Body"
        mediaDetail={{ type: 'document' }}
      />
    )

    const liveRegion = screen.getByTestId('content-selection-live-region')
    await waitFor(() =>
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    )
    const baselineAnnouncement = liveRegion.textContent

    rerender(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Body updated"
        mediaDetail={{ type: 'document' }}
      />
    )

    await waitFor(() => {
      expect(liveRegion.textContent).toBe(baselineAnnouncement)
      expect(liveRegion).toHaveTextContent('Showing Accessibility Item One')
    })
  })

  it('opens named read-along and annotation actions from keyboard selection events', async () => {
    const user = userEvent.setup()

    render(
      <ContentViewer
        selectedMedia={mediaOne}
        content="Keyboard selected sentence. Another sentence."
        mediaDetail={{ type: 'document' }}
        contentDisplayMode="plain"
      />
    )

    const contentRegion = screen.getByRole('region', { name: 'Media content' })
    for (let index = 0; index < 12 && document.activeElement !== contentRegion; index += 1) {
      await user.tab()
    }
    expect(contentRegion).toHaveFocus()

    const contentNode = screen.getByText(/Keyboard selected sentence/)
    selectText(contentNode, 'Keyboard selected sentence.')
    fireEvent.keyUp(contentNode)

    await waitFor(() => {
      expect(screen.getByTestId('media-selection-actions-popover')).toBeVisible()
    })
    expect(
      screen.getByRole('button', { name: 'Read selection' })
    ).toBeVisible()
    expect(screen.getByRole('button', { name: 'Read from here' })).toBeVisible()
    expect(
      screen.getByRole('button', { name: 'Read current section' })
    ).toBeVisible()
    expect(screen.getByRole('button', { name: 'Read full item' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Annotate' })).toBeVisible()
  })

  it('uses reduced-motion scrolling for active read-along segment follow mode', async () => {
    const scrollIntoView = vi.fn()
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView
    const originalMatchMedia = window.matchMedia

    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: vi.fn((query: string) => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
    HTMLElement.prototype.scrollIntoView = scrollIntoView
    const getBoundingClientRectSpy = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockImplementation(function getBoundingClientRectMock() {
        const element = this as HTMLElement
        if (element.dataset.readAlongSegmentId) {
          return makeRect({ left: 0, top: 1_000, width: 100, height: 24 })
        }
        if (element.dataset.testid === 'content-scroll-container') {
          return makeRect({ left: 0, top: 0, width: 320, height: 120 })
        }
        return makeRect({ left: 0, top: 0, width: 100, height: 24 })
      })

    mocks.readAlongState = {
      status: 'playing',
      scope: 'selection',
      activeSegmentId: '1101:0:sentence:0:15',
      activeIndex: 0,
      totalSegments: 1,
      error: null,
      cacheDisabled: false
    }

    try {
      render(
        <ContentViewer
          selectedMedia={mediaOne}
          content="First sentence. Second sentence."
          mediaDetail={{ type: 'document' }}
          contentDisplayMode="plain"
        />
      )

      await waitFor(() => {
        expect(scrollIntoView).toHaveBeenCalledWith({
          behavior: 'auto',
          block: 'center'
        })
      })
    } finally {
      getBoundingClientRectSpy.mockRestore()
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView
      Object.defineProperty(window, 'matchMedia', {
        configurable: true,
        value: originalMatchMedia
      })
    }
  })
})
