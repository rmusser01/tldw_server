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
let cacheSaveGate: Promise<void> | null = null

type DeferredAudio = {
  promise: Promise<{
    buffer: ArrayBuffer
    format: string
    mimeType: string
  }>
  resolve: (value: {
    buffer: ArrayBuffer
    format: string
    mimeType: string
  }) => void
}

const deferredAudio = (): DeferredAudio => {
  let resolve!: DeferredAudio['resolve']
  const promise = new Promise<{
    buffer: ArrayBuffer
    format: string
    mimeType: string
  }>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

const audioResult = (text: string) => ({
  buffer: new TextEncoder().encode(text).buffer,
  format: 'mp3',
  mimeType: 'audio/mpeg'
})

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
  applyBrowserSpeechSynthesisVoice: vi.fn(),
  resolveTtsProviderContext: vi.fn(async () => {
    eventLog.push(`provider:resolve:${providerContext.provider}`)
    return providerContext
  })
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getConfig: vi.fn(async () => ({
      serverUrl: 'http://127.0.0.1:8000/',
      apiKey: 'secret-api-key',
      authMode: 'single-user'
    }))
  }
}))

vi.mock('../media-read-along-cache-key', () => ({
  buildReadAlongCacheKey: vi.fn(async ({
    segmentId,
    segmentText,
    serverScope,
    settingsSignature
  }) => {
    eventLog.push(`cache-scope:${serverScope}`)
    eventLog.push(`cache-text:${segmentId}:${segmentText}`)
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
  saveMediaReadAlongAudioCacheEntry: vi.fn(async (entry, options) => {
    eventLog.push(`cache:save:${entry.id}`)
    if (cacheSaveGate) {
      await cacheSaveGate
    }
    if (options?.signal?.aborted || options?.shouldContinue?.() === false) {
      eventLog.push(`cache:skip:${entry.id}`)
      return false
    }
    cacheEntries.set(entry.id, {
      blob: entry.blob,
      mimeType: entry.mimeType,
      format: entry.format
    })
    return true
  })
}))

import {
  applyBrowserSpeechSynthesisVoice,
  resolveTtsProviderContext
} from '@/services/tts-provider'
import { tldwClient } from '@/services/tldw/TldwApiClient'
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
    cacheSaveGate = null
    providerContext = {
      provider: 'tldw',
      utterance: '',
      playbackSpeed: 1,
      supported: true,
      formatInfo: { requested: 'mp3', resolved: 'mp3', isFallback: false },
      normalizeText: (text: string) => text,
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
    vi.mocked(applyBrowserSpeechSynthesisVoice).mockReset()
    vi.mocked(tldwClient.getConfig).mockResolvedValue({
      serverUrl: 'http://127.0.0.1:8000/',
      apiKey: 'secret-api-key',
      authMode: 'single-user'
    })
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
      supported: true,
      normalizeText: (text: string) => text
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

  it('cleans up browser voice listeners when browser segments are replaced', async () => {
    providerContext = {
      provider: 'browser',
      utterance: 'First sentence.',
      playbackSpeed: 1,
      supported: true,
      browserVoiceName: 'Browser Voice',
      normalizeText: (text: string) => text
    }
    const cleanups: Array<ReturnType<typeof vi.fn>> = []
    vi.mocked(applyBrowserSpeechSynthesisVoice).mockImplementation(() => {
      const cleanup = vi.fn()
      cleanups.push(cleanup)
      return cleanup
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
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    act(() => {
      result.current.skip()
    })
    act(() => {
      result.current.stop()
    })

    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(2)
    expect(cleanups).toHaveLength(2)
    expect(cleanups[0]).toHaveBeenCalledTimes(1)
    expect(cleanups[1]).toHaveBeenCalledTimes(1)
  })

  it('targets retry and skip from the segment that failed while loading', async () => {
    providerContext.synthesize = vi.fn(async (text: string) => {
      eventLog.push(`synthesize:${text}`)
      if (text === 'Second sentence.') {
        throw new Error('second segment failed')
      }
      return audioResult(text)
    })
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    await completeCurrentAudio()

    await waitFor(() => expect(result.current.state.status).toBe('segment-error'))
    expect(result.current.state).toMatchObject({
      activeSegmentId: 'media-1:1:sentence:16:32',
      activeIndex: 1,
      error: 'second segment failed'
    })

    providerContext.synthesize = vi.fn(async (text: string) => {
      eventLog.push(`synthesize:${text}`)
      return audioResult(text)
    })
    act(() => {
      result.current.skip()
    })

    await waitFor(() =>
      expect(result.current.state.activeSegmentId).toBe('media-1:2:sentence:33:48')
    )
  })

  it('reuses an in-flight lookahead request when that segment becomes current', async () => {
    const pending = new Map<string, DeferredAudio>()
    providerContext.synthesize = vi.fn((text: string) => {
      eventLog.push(`synthesize:${text}`)
      if (text === 'Second sentence.') {
        const deferred = deferredAudio()
        pending.set(text, deferred)
        return deferred.promise
      }
      return Promise.resolve(audioResult(text))
    })
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    await waitFor(() => expect(pending.has('Second sentence.')).toBe(true))
    await completeCurrentAudio()
    await act(async () => {
      await Promise.resolve()
    })

    const secondCalls = vi
      .mocked(providerContext.synthesize)
      .mock.calls
      .filter(([text]) => text === 'Second sentence.')
    expect(secondCalls).toHaveLength(1)

    await act(async () => {
      pending.get('Second sentence.')!.resolve(audioResult('second'))
    })
    await waitFor(() =>
      expect(result.current.state.activeSegmentId).toBe('media-1:1:sentence:16:32')
    )
  })

  it('aborts stale lookahead work when retry restarts the current segment', async () => {
    const lookaheadSignals: AbortSignal[] = []
    const pending = new Map<string, DeferredAudio>()
    providerContext.synthesize = vi.fn((text: string, options?: { signal?: AbortSignal }) => {
      eventLog.push(`synthesize:${text}`)
      if (text === 'Second sentence.') {
        if (options?.signal) lookaheadSignals.push(options.signal)
        const deferred = deferredAudio()
        pending.set(text, deferred)
        return deferred.promise
      }
      return Promise.resolve(audioResult(text))
    })
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    await waitFor(() => expect(pending.has('Second sentence.')).toBe(true))

    act(() => {
      result.current.retry()
    })

    expect(lookaheadSignals[0]?.aborted).toBe(true)
  })

  it('splits over-cap provider requests while keeping the parent segment active', async () => {
    const longText = Array.from({ length: 900 }, (_, index) => `word${index}`).join(' ')
    providerContext.synthesize = vi.fn(async (text: string) => {
      eventLog.push(`synthesize:${text.length}`)
      return audioResult(text)
    })
    const { result } = setupHook({
      content: longText,
      displayContent: longText,
      selection: makeSelection({
        selectedText: longText,
        startSegmentId: `media-1:0:sentence:0:${longText.length}`,
        endSegmentId: `media-1:0:sentence:0:${longText.length}`,
        sourceStart: 0,
        sourceEnd: longText.length
      })
    })

    await act(async () => {
      await result.current.start('selection')
    })

    const synthesizeCalls = vi.mocked(providerContext.synthesize).mock.calls
    expect(synthesizeCalls.length).toBeGreaterThan(1)
    expect(synthesizeCalls.every(([text]) => String(text).length <= 4000)).toBe(true)
    expect(result.current.state).toMatchObject({
      status: 'playing',
      activeSegmentId: `media-1:0:sentence:0:${longText.length}`,
      activeIndex: 0,
      totalSegments: 1
    })
  })

  it('uses session-normalized text for generated synthesis and cache hashing', async () => {
    providerContext.normalizeText = vi.fn((text: string) => `normalized:${text}`)
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('selection')
    })

    expect(providerContext.normalizeText).toHaveBeenCalledWith('First sentence.')
    expect(providerContext.synthesize).toHaveBeenCalledWith(
      'normalized:First sentence.',
      expect.any(Object)
    )
    expect(eventLog).toContain(
      'cache-text:media-1:0:sentence:0:15:normalized:First sentence.'
    )
  })

  it('uses active server identity in generated-audio cache scope without secrets', async () => {
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('selection')
    })

    const scopeEntries = eventLog.filter((entry) => entry.startsWith('cache-scope:'))
    expect(scopeEntries[0]).toContain('http://127.0.0.1:8000')
    expect(scopeEntries[0]).not.toContain('secret-api-key')
    expect(scopeEntries[0]).not.toBe('cache-scope:media-read-along')
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
      supported: true,
      normalizeText: (text: string) => text
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

  it('ignores a pending generated segment when skip starts the next segment', async () => {
    const pending = new Map<string, DeferredAudio>()
    providerContext.synthesize = vi.fn((text: string) => {
      const deferred = deferredAudio()
      pending.set(text, deferred)
      return deferred.promise
    })
    const { result } = setupHook()

    act(() => {
      void result.current.start('full-item')
    })
    await waitFor(() => expect(pending.has('First sentence.')).toBe(true))

    act(() => {
      result.current.skip()
    })
    await waitFor(() => expect(pending.has('Second sentence.')).toBe(true))
    await act(async () => {
      pending.get('Second sentence.')!.resolve(audioResult('second'))
    })
    await waitFor(() =>
      expect(result.current.state.activeSegmentId).toBe('media-1:1:sentence:16:32')
    )

    await act(async () => {
      pending.get('First sentence.')!.resolve(audioResult('first'))
    })

    expect(result.current.state.activeSegmentId).toBe('media-1:1:sentence:16:32')
    expect(audioInstances).toHaveLength(1)
    expect(audioInstances[0]?.src).toBe('blob:read-along-0')
  })

  it('does not cache or play generated audio that resolves after stop', async () => {
    const deferred = deferredAudio()
    providerContext.synthesize = vi.fn(() => deferred.promise)
    const { result } = setupHook()

    act(() => {
      void result.current.start('selection')
    })
    await waitFor(() => expect(providerContext.synthesize).toHaveBeenCalledTimes(1))

    act(() => {
      result.current.stop()
    })
    await act(async () => {
      deferred.resolve(audioResult('stopped'))
    })

    expect(saveMediaReadAlongAudioCacheEntry).not.toHaveBeenCalled()
    expect(audioInstances).toHaveLength(0)
    expect(result.current.state.status).toBe('stopped')
  })

  it('rejects a generated cache save that is cancelled while the write is pending', async () => {
    let releaseSave!: () => void
    cacheSaveGate = new Promise<void>((resolve) => {
      releaseSave = resolve
    })
    const { result } = setupHook()

    act(() => {
      void result.current.start('selection')
    })
    await waitFor(() => expect(saveMediaReadAlongAudioCacheEntry).toHaveBeenCalledTimes(1))

    act(() => {
      result.current.stop()
    })
    await act(async () => {
      releaseSave()
      await cacheSaveGate
    })

    expect(cacheEntries.size).toBe(0)
    expect(eventLog).toContain('cache:skip:cache:media-1:0:sentence:0:15:tldw|||1|mp3|')
    expect(audioInstances).toHaveLength(0)
    expect(result.current.state.status).toBe('stopped')
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
      normalizeText: (text: string) => text,
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
      normalizeText: (text: string) => text,
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
      normalizeText: (text: string) => text,
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
      normalizeText: (text: string) => text,
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

  it('ignores old browser utterance callbacks after skip cancels speech', async () => {
    providerContext = {
      provider: 'browser',
      utterance: 'First sentence.',
      playbackSpeed: 1,
      supported: true,
      normalizeText: (text: string) => text
    }
    const spoken: MockSpeechSynthesisUtterance[] = []
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: {
        speak: vi.fn((utterance: MockSpeechSynthesisUtterance) => {
          spoken.push(utterance)
          eventLog.push(`speech:speak:${utterance.text}`)
        }),
        cancel: vi.fn(),
        pause: vi.fn(),
        resume: vi.fn()
      }
    })
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })
    expect(spoken[0]?.text).toBe('First sentence.')

    act(() => {
      result.current.skip()
    })
    expect(spoken[1]?.text).toBe('Second sentence.')

    act(() => {
      spoken[0]!.onend?.()
      spoken[0]!.onerror?.()
    })

    expect(result.current.state).toMatchObject({
      status: 'playing',
      activeSegmentId: 'media-1:1:sentence:16:32',
      error: null
    })
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(2)
  })

  it('invalidates browser utterance callbacks before synchronous cancel during skip', async () => {
    providerContext = {
      provider: 'browser',
      utterance: 'First sentence.',
      playbackSpeed: 1,
      supported: true,
      normalizeText: (text: string) => text
    }
    const spoken: MockSpeechSynthesisUtterance[] = []
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: {
        speak: vi.fn((utterance: MockSpeechSynthesisUtterance) => {
          spoken.push(utterance)
          eventLog.push(`speech:speak:${utterance.text}`)
        }),
        cancel: vi.fn(() => {
          spoken[0]?.onerror?.()
          spoken[0]?.onend?.()
        }),
        pause: vi.fn(),
        resume: vi.fn()
      }
    })
    const { result } = setupHook()

    await act(async () => {
      await result.current.start('full-item')
    })

    act(() => {
      result.current.skip()
    })

    expect(result.current.state).toMatchObject({
      status: 'playing',
      activeSegmentId: 'media-1:1:sentence:16:32',
      error: null
    })
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(2)
    expect(spoken.map((utterance) => utterance.text)).toEqual([
      'First sentence.',
      'Second sentence.'
    ])
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
