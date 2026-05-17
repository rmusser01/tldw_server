import React from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ContentViewer } from '../ContentViewer'
import { useMediaReadingProgress } from '@/hooks/useMediaReadingProgress'

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

const cacheEntries = new Map<string, { blob: Blob; mimeType: string; format: string }>()
const audioInstances: MockAudio[] = []
const originalCreateObjectUrlDescriptor = Object.getOwnPropertyDescriptor(
  URL,
  'createObjectURL'
)
const originalRevokeObjectUrlDescriptor = Object.getOwnPropertyDescriptor(
  URL,
  'revokeObjectURL'
)

class MockAudio extends EventTarget {
  src: string
  currentTime = 0
  playbackRate = 1
  paused = true
  play = vi.fn(async () => {
    this.paused = false
  })
  pause = vi.fn(() => {
    this.paused = true
  })

  constructor(src = '') {
    super()
    this.src = src
    audioInstances.push(this)
  }
}

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | {
            defaultValue?: string
            count?: number
            index?: number
            total?: number
            timestamp?: string
          }
    ) => {
      if (typeof fallbackOrOptions === 'string') return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return fallbackOrOptions.defaultValue
          .replace('{{count}}', String(fallbackOrOptions.count ?? ''))
          .replace('{{index}}', String(fallbackOrOptions.index ?? ''))
          .replace('{{total}}', String(fallbackOrOptions.total ?? ''))
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

vi.mock('@/services/tts-provider', () => ({
  applyBrowserSpeechSynthesisVoice: vi.fn(),
  resolveTtsProviderContext: vi.fn(async () => ({
    provider: 'tldw',
    utterance: '',
    cacheSettings: {
      provider: 'tldw',
      model: 'tts-1',
      voice: 'alloy',
      speed: 1,
      format: 'mp3'
    },
    formatInfo: {
      requested: 'mp3',
      resolved: 'mp3',
      extension: 'mp3',
      mimeType: 'audio/mpeg'
    },
    playbackSpeed: 1,
    normalizeText: (text: string) => text,
    synthesize: vi.fn(async (text: string) => ({
      buffer: new TextEncoder().encode(text).buffer,
      format: 'mp3',
      mimeType: 'audio/mpeg'
    }))
  }))
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getConfig: vi.fn(async () => ({
      serverUrl: 'http://127.0.0.1:8000/',
      apiKey: 'test-api-key',
      authMode: 'single-user'
    }))
  }
}))

vi.mock('../read-along/media-read-along-cache-key', () => ({
  buildReadAlongCacheKey: vi.fn(async ({ segmentId, settingsSignature }) => ({
    id: `cache:${segmentId}:${settingsSignature}`,
    mediaId: 'media-read-along-test',
    mediaKind: 'media',
    segmentId,
    settingsSignature,
    textHash: `hash:${segmentId}`
  })),
  buildTtsSettingsSignature: vi.fn(() => 'mock-settings')
}))

vi.mock('../read-along/media-read-along-cache', () => ({
  getMediaReadAlongAudioCacheEntry: vi.fn(async (id: string) => {
    const entry = cacheEntries.get(id)
    return entry
      ? {
          id,
          createdAt: 1,
          lastUsedAt: 1,
          mediaId: 'media-read-along-test',
          mediaKind: 'media',
          segmentId: id,
          settingsSignature: 'mock-settings',
          textHash: 'mock',
          blob: entry.blob,
          mimeType: entry.mimeType,
          format: entry.format,
          sizeBytes: entry.blob.size
        }
      : undefined
  }),
  saveMediaReadAlongAudioCacheEntry: vi.fn(async (entry) => {
    cacheEntries.set(entry.id, {
      blob: entry.blob,
      mimeType: entry.mimeType,
      format: entry.format
    })
    return true
  })
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
  id: 930,
  title: 'Read-along target',
  raw: {},
  meta: {
    type: 'document'
  }
}

const videoMedia = {
  ...selectedMedia,
  id: 931,
  title: 'Video read-along target',
  raw: {
    has_original_file: true
  },
  meta: {
    type: 'video'
  }
}

const renderViewer = (
  overrides: Partial<React.ComponentProps<typeof ContentViewer>> = {}
) =>
  render(
    <ContentViewer
      selectedMedia={selectedMedia}
      content={'First sentence. Second sentence.'}
      mediaDetail={{ type: 'document' }}
      contentDisplayMode="plain"
      {...overrides}
    />
  )

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
  fireEvent.mouseUp(node)
}

const selectTextInside = (text: string) => {
  const candidates = screen.getAllByText((_content, element) =>
    Boolean(element?.textContent?.includes(text))
  )
  const node =
    candidates.find(
      (element) =>
        !Array.from(element.children).some((child) =>
          child.textContent?.includes(text)
        )
    ) || candidates[0]
  selectText(node, text)
  return node
}

const restoreUrlObjectUrlHelpers = () => {
  if (originalCreateObjectUrlDescriptor) {
    Object.defineProperty(URL, 'createObjectURL', originalCreateObjectUrlDescriptor)
  } else {
    delete (URL as any).createObjectURL
  }
  if (originalRevokeObjectUrlDescriptor) {
    Object.defineProperty(URL, 'revokeObjectURL', originalRevokeObjectUrlDescriptor)
  } else {
    delete (URL as any).revokeObjectURL
  }
}

describe('ContentViewer read-along integration', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    cacheEntries.clear()
    audioInstances.length = 0
    mocks.bgRequest.mockReset()
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (String(request?.path || '').endsWith('/file')) {
        return new Uint8Array([1, 2, 3]).buffer
      }
      return {}
    })
    vi.mocked(useMediaReadingProgress).mockReturnValue({
      saveProgress: vi.fn(),
      clearProgress: vi.fn(),
      progressPercent: null
    })
    vi.stubGlobal('Audio', MockAudio)
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:read-along-test')
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn()
    })
  })

  afterEach(() => {
    window.getSelection()?.removeAllRanges()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    restoreUrlObjectUrlHelpers()
  })

  it('shows no read-along UI before content selection', () => {
    renderViewer()

    expect(screen.queryByTestId('media-selection-actions-popover')).not.toBeInTheDocument()
    expect(screen.queryByTestId('media-read-along-transport')).not.toBeInTheDocument()
  })

  it('shows read-along and annotation actions in the selection popover', async () => {
    renderViewer()

    selectTextInside('First sentence.')

    await waitFor(() => {
      expect(screen.getByTestId('media-selection-actions-popover')).toBeInTheDocument()
    })
    expect(screen.getByTestId('media-selection-action-read-selection')).toHaveTextContent(
      'Read selection'
    )
    expect(screen.getByTestId('media-selection-action-read-from-here')).toHaveTextContent(
      'Read from here'
    )
    expect(
      screen.getByTestId('media-selection-action-read-current-section')
    ).toHaveTextContent('Read current section')
    expect(screen.getByTestId('media-selection-action-read-full-item')).toHaveTextContent(
      'Read full item'
    )
    expect(screen.getByTestId('media-selection-action-annotate')).toHaveTextContent(
      'Annotate'
    )
  })

  it('starts reading a selection and keeps the inline transport visible after selection clears', async () => {
    renderViewer()

    selectTextInside('First sentence.')
    fireEvent.click(await screen.findByTestId('media-selection-action-read-selection'))

    await waitFor(() => {
      expect(screen.getByTestId('media-read-along-transport')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('media-selection-actions-popover')).not.toBeInTheDocument()
    expect(window.getSelection()?.rangeCount).toBe(0)
    expect(audioInstances.length).toBeGreaterThan(0)
  })

  it('marks the active readable segment in plain content', async () => {
    renderViewer()

    selectTextInside('First sentence.')
    fireEvent.click(await screen.findByTestId('media-selection-action-read-selection'))

    await waitFor(() => {
      expect(
        document.querySelector('[data-read-along-active="true"]')
      ).toBeInTheDocument()
    })
    expect(document.querySelector('[data-read-along-active="true"]')).toHaveTextContent(
      'First sentence.'
    )
  })

  it('hides transient read-along UI after stop', async () => {
    renderViewer()

    selectTextInside('First sentence.')
    fireEvent.click(await screen.findByTestId('media-selection-action-read-selection'))
    await screen.findByTestId('media-read-along-transport')

    fireEvent.click(screen.getByTestId('media-read-along-stop'))

    await waitFor(() => {
      expect(screen.queryByTestId('media-read-along-transport')).not.toBeInTheDocument()
    })
    expect(document.querySelector('[data-read-along-active="true"]')).not.toBeInTheDocument()
    expect(screen.queryByTestId('media-selection-actions-popover')).not.toBeInTheDocument()
  })

  it('pauses embedded media preview when generated read-along playback starts', async () => {
    renderViewer({
      selectedMedia: videoMedia,
      mediaDetail: {
        type: 'video',
        has_original_file: true
      }
    })

    const video = await screen.findByTestId('embedded-video-player')
    let paused = false
    const pause = vi.fn(() => {
      paused = true
    })
    Object.defineProperty(video, 'paused', {
      configurable: true,
      get: () => paused
    })
    Object.defineProperty(video, 'pause', {
      configurable: true,
      value: pause
    })

    selectTextInside('First sentence.')
    fireEvent.click(await screen.findByTestId('media-selection-action-read-selection'))

    await waitFor(() => {
      expect(pause).toHaveBeenCalled()
    })
  })

  it('starts markdown and html fallback playback without mutating rich HTML', async () => {
    const { unmount } = renderViewer({
      content: 'Markdown fallback sentence.',
      contentDisplayMode: 'markdown'
    })

    selectTextInside('Markdown fallback sentence.')
    fireEvent.click(await screen.findByTestId('media-selection-action-read-selection'))

    await waitFor(() => {
      expect(screen.getByTestId('media-read-along-transport')).toBeInTheDocument()
    })

    unmount()
    window.getSelection()?.removeAllRanges()

    renderViewer({
      content: '<p>HTML fallback sentence.</p>',
      contentDisplayMode: 'html',
      allowRichRendering: true
    })

    const htmlText = screen.getByText('HTML fallback sentence.')
    const htmlRegion = htmlText.closest('[role="region"]')
    expect(htmlRegion?.innerHTML).not.toContain('data-read-along-segment-id')

    selectText(htmlText, 'HTML fallback sentence.')
    fireEvent.click(await screen.findByTestId('media-selection-action-read-selection'))

    await waitFor(() => {
      expect(screen.getByTestId('media-read-along-transport')).toBeInTheDocument()
    })
    expect(htmlRegion?.innerHTML).not.toContain('data-read-along-segment-id')
  })
})
