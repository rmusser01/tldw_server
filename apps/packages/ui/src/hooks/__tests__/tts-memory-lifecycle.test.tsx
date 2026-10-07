import React from 'react'
import { act, renderHook, cleanup } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { useTtsPlayground } from '@/hooks/useTtsPlayground'
import { useDocumentTTS } from '@/hooks/document-workspace/useDocumentTTS'

type SynthesizedAudio = { buffer: ArrayBuffer; mimeType: string; format: string }
type AudioResponse = { ok: boolean; blob: () => Promise<Blob> }

const probe = vi.hoisted(() => ({ synthesize: vi.fn(), allocated: new Set<string>(), serial: 0 }))
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock('@/hooks/useAntdNotification', () => ({
  useAntdNotification: () => ({ warning: vi.fn(), info: vi.fn(), error: vi.fn() })
}))
vi.mock('@/utils/tts', () => ({ splitMessageContent: (text: string) => [text] }))
vi.mock('@/services/tts-provider', () => ({
  resolveTtsProviderContext: async (text: string) => ({
    provider: 'tldw',
    supported: true,
    utterance: text,
    synthesize: probe.synthesize
  })
}))
vi.mock('@/services/tldw/audio-voices', () => ({ fetchTldwVoices: async () => [] }))
vi.mock('@/services/tts', () => ({
  DEFAULT_TLDW_TTS_VOICE: 'Bella',
  getTldwTTSModel: async () => 'mock',
  getTldwTTSVoice: async () => 'Bella'
}))

class MockAudio {
  paused = true
  ended = false
  duration = 1
  currentTime = 0
  volume = 1
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  onplay: (() => void) | null = null
  onpause: (() => void) | null = null
  ontimeupdate: (() => void) | null = null
  constructor(public src: string) {}
  pause() {
    this.paused = true
    this.onpause?.()
  }
  play() {
    this.paused = false
    this.onplay?.()
    return Promise.resolve()
  }
}

beforeEach(() => {
  probe.allocated.clear()
  probe.synthesize.mockResolvedValue({
    buffer: new Uint8Array([1, 2, 3]).buffer,
    mimeType: 'audio/mpeg',
    format: 'mp3'
  })
  vi.spyOn(URL, 'createObjectURL').mockImplementation(() => {
    const url = `blob:review-${++probe.serial}`
    probe.allocated.add(url)
    return url
  })
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation((url) => {
    probe.allocated.delete(url)
  })
  vi.stubGlobal('Audio', MockAudio)
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function documentWrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: React.PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

it('a generated Speech/TTS segment URL is released after unmount', async () => {
  const hook = renderHook(() => useTtsPlayground())
  await act(async () => {
    await hook.result.current.generateSegments('First speech.')
  })
  hook.unmount()
  expect(probe.allocated.size).toBe(0)
})

it('in-flight Speech/TTS generation cannot create a URL after unmount', async () => {
  let finish!: (value: SynthesizedAudio) => void
  probe.synthesize.mockImplementation(
    () =>
      new Promise<SynthesizedAudio>((resolve) => {
        finish = resolve
      })
  )
  const hook = renderHook(() => useTtsPlayground())
  let pending!: ReturnType<ReturnType<typeof useTtsPlayground>['generateSegments']>
  await act(async () => {
    pending = hook.result.current.generateSegments('Pending speech.')
    await Promise.resolve()
  })
  hook.unmount()
  await act(async () => {
    finish({ buffer: new Uint8Array([1, 2, 3]).buffer, mimeType: 'audio/mpeg', format: 'mp3' })
    await pending
  })
  expect(probe.allocated.size).toBe(0)
})

it('Stop retires a pending Document TTS response', async () => {
  let finish!: (value: AudioResponse) => void
  const response = new Promise<AudioResponse>((resolve) => {
    finish = resolve
  })
  vi.stubGlobal(
    'fetch',
    vi.fn(() => response)
  )
  const hook = renderHook(() => useDocumentTTS(), { wrapper: documentWrapper() })
  let pending!: Promise<void>
  await act(async () => {
    pending = hook.result.current.speak('Pending document speech.')
    await Promise.resolve()
  })
  act(() => hook.result.current.stop())
  await act(async () => {
    finish({ ok: true, blob: async () => new Blob(['audio']) })
    await pending
  })
  expect(hook.result.current.state.isPlaying).toBe(false)
  expect(probe.allocated.size).toBe(0)
})

it('Document TTS discards audio after unmount', async () => {
  let finish!: (value: AudioResponse) => void
  const response = new Promise<AudioResponse>((resolve) => {
    finish = resolve
  })
  vi.stubGlobal(
    'fetch',
    vi.fn(() => response)
  )
  const hook = renderHook(() => useDocumentTTS(), { wrapper: documentWrapper() })
  let pending!: Promise<void>
  await act(async () => {
    pending = hook.result.current.speak('Pending document speech.')
    await Promise.resolve()
  })
  hook.unmount()
  await act(async () => {
    finish({ ok: true, blob: async () => new Blob(['audio']) })
    await pending
  })
  expect(probe.allocated.size).toBe(0)
})

it('releases replaced and explicitly cleared playground URLs', async () => {
  const hook = renderHook(() => useTtsPlayground())
  await act(async () => {
    await hook.result.current.generateSegments('First.')
  })
  await act(async () => {
    await hook.result.current.generateSegments('Second.')
  })
  expect(probe.allocated.size).toBe(1)
  act(() => hook.result.current.clearSegments())
  expect(probe.allocated.size).toBe(0)
})

it('owns external segments and revokes late updates after unmount', () => {
  const hook = renderHook(() => useTtsPlayground())
  const url = URL.createObjectURL(new Blob(['external']))
  act(() => hook.result.current.setSegments([{ id: 'external', index: 0, text: 'External', url }]))
  hook.unmount()
  const late = URL.createObjectURL(new Blob(['late']))
  act(() => hook.result.current.setSegments([{ id: 'late', index: 0, text: 'Late', url: late }]))
  expect(probe.allocated.size).toBe(0)
})
