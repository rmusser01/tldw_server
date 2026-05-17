import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { TtsProviderContext } from '@/services/tts-provider'

import type { ReadAlongSelection } from '../types'

const cacheEntries = new Map<string, { blob: Blob; mimeType: string; format: string }>()
const abortControllers: AbortController[] = []
const audioInstances: MockAudio[] = []
const eventLog: string[] = []

let providerContext: TtsProviderContext
let objectUrlCounter = 0
let playRejects = false

class TrackingAbortController extends AbortController {
  constructor() {
    super()
    abortControllers.push(this)
  }
}

class MockAudio extends EventTarget {
  src: string
  currentTime = 0
  playbackRate = 1
  paused = true
  play = vi.fn(async () => {
    eventLog.push(`audio:play:${this.src}`)
    if (playRejects) {
      throw new Error('autoplay blocked')
    }
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

class MockSpeechSynthesisUtterance extends EventTarget {
  text: string
  rate = 1
  onend: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(text: string) {
    super()
    this.text = text
  }
}

vi.mock('@/services/tts-provider', () => ({
  resolveTtsProviderContext: vi.fn(async () => {
    eventLog.push(`provider:resolve:${providerContext.provider}`)
    return providerContext
  })
}))

vi.mock('../media-read-along-cache-key', () => ({
  buildReadAlongCacheKey: vi.fn(async ({ segmentId, settingsSignature }) => {
    eventLog.push(`cache-key:${segmentId}:${settingsSignature}`)
    return {
      id: `cache:${segmentId}:${settingsSignature}`,
      mediaId: 'media-1',
      mediaKind: 'video',
      segmentId,
      settingsSignature,
      textHash: `hash:${segmentId}`
    }
  }),
  buildTtsSettingsSignature: vi.fn((settings) => {
    const signature = [
      settings.provider,
      settings.model ?? '',
      settings.voice ?? '',
      settings.speed ?? '',
      settings.format ?? '',
      settings.language ?? ''
    ].join('|')
    eventLog.push(`settings:${signature}`)
    return signature
  })
}))

vi.mock('../media-read-along-cache', () => ({
  getMediaReadAlongAudioCacheEntry: vi.fn(async (id: string) => {
    eventLog.push(`cache:get:${id}`)
    const entry = cacheEntries.get(id)
    return entry
      ? {
          id,
          createdAt: 1,
          lastUsedAt: 1,
          mediaId: 'media-1',
          mediaKind: 'video',
          segmentId: id,
          settingsSignature: 'mock',
          textHash: 'mock',
          blob: entry.blob,
          mimeType: entry.mimeType,
          format: entry.format,
          sizeBytes: entry.blob.size
        }
      : undefined
  }),
  saveMediaReadAlongAudioCacheEntry: vi.fn(async (entry) => {
    eventLog.push(`cache:save:${entry.id}`)
    cacheEntries.set(entry.id, {
      blob: entry.blob,
      mimeType: entry.mimeType,
      format: entry.format
    })
    return true
  })
}))

import { resolveTtsProviderContext } from '@/services/tts-provider'
import {
  getMediaReadAlongAudioCacheEntry,
  saveMediaReadAlongAudioCacheEntry
} from '../media-read-along-cache'
import { buildReadAlongCacheKey } from '../media-read-along-cache-key'
import { useMediaReadAlongSession } from '../useMediaReadAlongSession'

const makeSelection = (overrides: Partial<ReadAlongSelection> = {}): ReadAlongSelection => ({
  selectedText: 'First sentence.',
  anchorRect: new DOMRect(0, 0, 10, 10),
  startSegmentId: 'media-1:0:sentence:0:15',
  endSegmentId: 'media-1:0:sentence:0:15',
  sourceStart: 0,
  sourceEnd: 15,
  mappingConfidence: 'exact',
  ...overrides
})

const content = [
  'First sentence.',
  'Second sentence.',
  'Third sentence.',
  'Fourth sentence.',
  'Fifth sentence.',
  'Sixth sentence.',
  'Seventh sentence.',
  'Eighth sentence.'
].join(' ')

const setupHook = (overrides: Partial<Parameters<typeof useMediaReadAlongSession>[0]> = {}) => {
  const contentBody = document.createElement('div')
  const scrollContainer = document.createElement('div')
  const embeddedMedia = document.createElement('video')
  document.body.append(contentBody, scrollContainer, embeddedMedia)

  return renderHook((props: Partial<Parameters<typeof useMediaReadAlongSession>[0]>) =>
    useMediaReadAlongSession({
      mediaId: 'media-1',
      mediaKind: 'video',
      content,
      displayContent: 'First sentence. Second sentence.',
      renderMode: 'plain',
      hideTranscriptTimings: false,
      selection: makeSelection(),
      contentBodyRef: { current: contentBody },
      contentScrollContainerRef: { current: scrollContainer },
      embeddedMediaRef: { current: embeddedMedia },
      ...overrides,
      ...props
    })
  )
}

const completeCurrentAudio = async () => {
  const audio = audioInstances.at(-1)
  expect(audio).toBeDefined()
  await act(async () => {
    audio!.dispatchEvent(new Event('ended'))
  })
}

describe('useMediaReadAlongSession', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    cacheEntries.clear()
    abortControllers.length = 0
    audioInstances.length = 0
    eventLog.length = 0
    objectUrlCounter = 0
    playRejects = false
    providerContext = {
      provider: 'tldw',
      utterance: '',
      playbackSpeed: 1,
      supported: true,
      formatInfo: { requested: 'mp3', resolved: 'mp3', isFallback: false },
      synthesize: vi.fn(async (text: string) => {
        eventLog.push(`synthesize:${text}`)
        return {
          buffer: new TextEncoder().encode(text).buffer,
          format: 'mp3',
          mimeType: 'audio/mpeg'
        }
      })
    }
    vi.clearAllMocks()
    vi.stubGlobal('AbortController', TrackingAbortController)
    vi.stubGlobal('Audio', MockAudio)
    vi.stubGlobal('SpeechSynthesisUtterance', MockSpeechSynthesisUtterance)
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => `blob:read-along-${objectUrlCounter++}`)
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn()
    })
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: {
        speak: vi.fn((utterance: MockSpeechSynthesisUtterance) => {
          eventLog.push(`speech:speak:${utterance.text}`)
        }),
        cancel: vi.fn(),
        pause: vi.fn(),
        resume: vi.fn()
      }
    })
  })

  it('start("selection") resolves a queue and plays cached audio before generating lookahead', async () => {
    cacheEntries.set('cache:media-1:0:sentence:0:15:tldw|||1|mp3|', {
      blob: new Blob(['cached'], { type: 'audio/mpeg' }),
      mimeType: 'audio/mpeg',
      format: 'mp3'
    })
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('selection')
    })

    await waitFor(() => expect(audioInstances[0]?.play).toHaveBeenCalledTimes(1))
    expect(result.current.state).toMatchObject({
      status: 'playing',
      scope: 'selection',
      activeSegmentId: 'media-1:0:sentence:0:15',
      activeIndex: 0,
      totalSegments: 1
    })
    expect(eventLog.indexOf('audio:play:blob:read-along-0')).toBeGreaterThan(
      eventLog.indexOf('cache:get:cache:media-1:0:sentence:0:15:tldw|||1|mp3|')
    )
    expect(eventLog.findIndex((entry) => entry.startsWith('synthesize:'))).toBe(-1)
  })

  it('start("from-here") queues beyond the rendered window', async () => {
    const { result } = setupHook({
      selection: makeSelection({
        selectedText: 'Third sentence.',
        startSegmentId: 'media-1:2:sentence:33:48',
        endSegmentId: 'media-1:2:sentence:33:48',
        sourceStart: 33,
        sourceEnd: 48
      })
    })

    await act(async () => {
      await result.current.start('from-here')
    })

    await waitFor(() => expect(result.current.state.totalSegments).toBe(6))
    expect(result.current.state.activeSegmentId).toBe('media-1:2:sentence:33:48')
    expect(providerContext.synthesize).toHaveBeenCalledWith(
      'Third sentence.',
      expect.any(Object)
    )
  })

  it('lookahead prefetches 3 to 5 segments, not the full item', async () => {
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })

    await waitFor(() =>
      expect((providerContext.synthesize as ReturnType<typeof vi.fn>).mock.calls.length)
        .toBeGreaterThanOrEqual(4)
    )
    expect((providerContext.synthesize as ReturnType<typeof vi.fn>).mock.calls.length)
      .toBeLessThanOrEqual(6)
    expect((providerContext.synthesize as ReturnType<typeof vi.fn>).mock.calls.length)
      .toBeLessThan(8)
  })

  it('browser TTS provider uses window.speechSynthesis.speak() and does not touch generated-audio cache', async () => {
    providerContext = {
      provider: 'browser',
      utterance: 'First sentence.',
      playbackSpeed: 1.25,
      supported: true
    }
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('selection')
    })

    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1)
    expect(getMediaReadAlongAudioCacheEntry).not.toHaveBeenCalled()
    expect(saveMediaReadAlongAudioCacheEntry).not.toHaveBeenCalled()
    expect(buildReadAlongCacheKey).not.toHaveBeenCalled()
  })

  it('stop aborts current and lookahead AbortControllers', async () => {
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    await waitFor(() => expect(abortControllers.length).toBeGreaterThanOrEqual(2))

    act(() => {
      result.current.stop()
    })

    expect(abortControllers.every((controller) => controller.signal.aborted)).toBe(true)
    expect(result.current.state.status).toBe('stopped')
  })

  it('stop cancels browser SpeechSynthesis when browser provider is active', async () => {
    providerContext = {
      provider: 'browser',
      utterance: 'First sentence.',
      playbackSpeed: 1,
      supported: true
    }
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('selection')
    })
    act(() => {
      result.current.stop()
    })

    expect(window.speechSynthesis.cancel).toHaveBeenCalledTimes(1)
  })

  it('media/content change stops and suppresses stale completions', async () => {
    let finishSynthesis!: (value: {
      buffer: ArrayBuffer
      format: string
      mimeType: string
    }) => void
    providerContext.synthesize = vi.fn(
      () =>
        new Promise((resolve) => {
          finishSynthesis = resolve
        })
    )
    const { result, rerender } = setupHook()

    await act(async () => {
      void result.current.start('selection')
    })
    rerender({ mediaId: 'media-2' })
    await act(async () => {
      finishSynthesis({
        buffer: new ArrayBuffer(8),
        format: 'mp3',
        mimeType: 'audio/mpeg'
      })
    })

    expect(audioInstances).toHaveLength(0)
    expect(result.current.state).toMatchObject({
      status: 'stopped',
      activeSegmentId: null
    })
  })

  it('settings are captured at session start and do not mutate mid-session', async () => {
    const firstSynthesize = vi.fn(async (text: string) => ({
      buffer: new TextEncoder().encode(`first:${text}`).buffer,
      format: 'mp3',
      mimeType: 'audio/mpeg'
    }))
    const secondSynthesize = vi.fn(async (text: string) => ({
      buffer: new TextEncoder().encode(`second:${text}`).buffer,
      format: 'wav',
      mimeType: 'audio/wav'
    }))
    providerContext = {
      provider: 'tldw',
      utterance: '',
      playbackSpeed: 1,
      supported: true,
      formatInfo: { requested: 'mp3', resolved: 'mp3', isFallback: false },
      synthesize: firstSynthesize
    }
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    providerContext = {
      provider: 'tldw',
      utterance: '',
      playbackSpeed: 1,
      supported: true,
      formatInfo: { requested: 'wav', resolved: 'wav', isFallback: false },
      synthesize: secondSynthesize
    }
    await completeCurrentAudio()

    await waitFor(() => expect(firstSynthesize.mock.calls.length).toBeGreaterThan(1))
    expect(resolveTtsProviderContext).toHaveBeenCalledTimes(1)
    expect(secondSynthesize).not.toHaveBeenCalled()
    expect(eventLog.some((entry) => entry.includes('tldw|||1|mp3|'))).toBe(true)
    expect(eventLog.some((entry) => entry.includes('tldw|||1|wav|'))).toBe(false)
  })

  it('uses effective model and voice in generated-audio cache signatures', async () => {
    providerContext = {
      provider: 'openai',
      utterance: '',
      playbackSpeed: 1,
      supported: true,
      cacheSettings: {
        provider: 'openai',
        model: 'tts-model-a',
        voice: 'voice-a',
        speed: 1,
        format: 'mp3'
      },
      synthesize: vi.fn(async (text: string) => ({
        buffer: new TextEncoder().encode(text).buffer,
        format: 'mp3',
        mimeType: 'audio/mpeg'
      }))
    }
    const first = setupHook()

    await act(async () => {
      await first.result.current.start('selection')
    })
    first.unmount()

    providerContext = {
      provider: 'openai',
      utterance: '',
      playbackSpeed: 1,
      supported: true,
      cacheSettings: {
        provider: 'openai',
        model: 'tts-model-b',
        voice: 'voice-b',
        speed: 1,
        format: 'mp3'
      },
      synthesize: vi.fn(async (text: string) => ({
        buffer: new TextEncoder().encode(text).buffer,
        format: 'mp3',
        mimeType: 'audio/mpeg'
      }))
    }
    const second = setupHook()

    await act(async () => {
      await second.result.current.start('selection')
    })

    expect(eventLog).toContain('settings:openai|tts-model-a|voice-a|1|mp3|')
    expect(eventLog).toContain('settings:openai|tts-model-b|voice-b|1|mp3|')
    expect(eventLog).toContain(
      'cache-key:media-1:0:sentence:0:15:openai|tts-model-a|voice-a|1|mp3|'
    )
    expect(eventLog).toContain(
      'cache-key:media-1:0:sentence:0:15:openai|tts-model-b|voice-b|1|mp3|'
    )
  })

  it('audio.play() rejection enters segment-error', async () => {
    playRejects = true
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('selection')
    })

    await waitFor(() => expect(result.current.state.status).toBe('segment-error'))
    expect(result.current.state.error).toContain('autoplay blocked')
  })

  it('starting read-along pauses embeddedMediaRef.current if it is playing', async () => {
    const embeddedMedia = document.createElement('video')
    const pause = vi.fn()
    Object.defineProperty(embeddedMedia, 'paused', {
      configurable: true,
      value: false
    })
    Object.defineProperty(embeddedMedia, 'pause', {
      configurable: true,
      value: pause
    })
    const { result } = setupHook({
      embeddedMediaRef: { current: embeddedMedia }
    })

    await act(async () => {
      await result.current.start('selection')
    })

    expect(pause).toHaveBeenCalledTimes(1)
  })
})
