import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/services/tts', () => ({
  getOpenAITTSModel: vi.fn(async () => 'tts-1'),
  getOpenAITTSVoice: vi.fn(async () => 'alloy'),
  getVoice: vi.fn(async () => 'Browser Voice'),
  getElevenLabsApiKey: vi.fn(async () => ''),
  getElevenLabsModel: vi.fn(async () => ''),
  getElevenLabsVoiceId: vi.fn(async () => ''),
  getRemoveReasoningTagTTS: vi.fn(async () => false),
  getSpeechPlaybackSpeed: vi.fn(async () => 1),
  getTTSProvider: vi.fn(async () => 'browser'),
  getTldwTTSModel: vi.fn(async () => 'kokoro'),
  getTldwTTSResponseFormat: vi.fn(async () => 'mp3'),
  getTldwTTSSpeed: vi.fn(async () => 1),
  getTldwTTSVoice: vi.fn(async () => 'af_heart'),
  isSSMLEnabled: vi.fn(async () => false),
  isSupportedTldwTtsResponseFormat: vi.fn(() => true),
  normalizeTldwTtsResponseFormat: vi.fn((format: string) => format)
}))

vi.mock('@/services/tts-providers', () => ({
  TTS_PROVIDER_VALUES: ['browser', 'elevenlabs', 'openai', 'tldw']
}))

vi.mock('@/utils/markdown-to-ssml', () => ({
  markdownToSSML: vi.fn((text: string) => text)
}))

vi.mock('@/libs/reasoning', () => ({
  removeReasoning: vi.fn((text: string) => text)
}))

vi.mock('@/utils/markdown-to-text', () => ({
  markdownToText: vi.fn((text: string) => text)
}))

vi.mock('@/services/elevenlabs', () => ({
  generateSpeech: vi.fn()
}))

vi.mock('@/services/openai-tts', () => ({
  generateOpenAITTS: vi.fn()
}))

vi.mock('@/utils/provider-registry', () => ({
  inferProviderFromModel: vi.fn(() => 'kokoro')
}))

vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    synthesizeSpeech: vi.fn(async () => new ArrayBuffer(8))
  }
}))

import { tldwClient } from '@/services/tldw/TldwApiClient'
import { generateSpeech } from '@/services/elevenlabs'
import { generateOpenAITTS } from '@/services/openai-tts'
import { markdownToText } from '@/utils/markdown-to-text'
import {
  getElevenLabsApiKey,
  getElevenLabsModel,
  getElevenLabsVoiceId,
  getOpenAITTSModel,
  getOpenAITTSVoice,
  getVoice,
  isSSMLEnabled
} from '@/services/tts'
import {
  applyBrowserSpeechSynthesisVoice,
  resolveTtsProviderContext
} from '../tts-provider'

describe('tts provider read-along synthesis', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('passes abort signals through tldw synthesis', async () => {
    const signal = new AbortController().signal
    const context = await resolveTtsProviderContext('hello', { provider: 'tldw' })

    await context.synthesize?.('hello', { signal })

    expect(tldwClient.synthesizeSpeech).toHaveBeenCalledWith(
      'hello',
      expect.objectContaining({ signal })
    )
  })

  it('captures OpenAI model and voice when provider context is resolved', async () => {
    vi.mocked(getOpenAITTSModel).mockResolvedValueOnce('model-a')
    vi.mocked(getOpenAITTSVoice).mockResolvedValueOnce('voice-a')
    vi.mocked(generateOpenAITTS).mockResolvedValueOnce(new ArrayBuffer(8))

    const context = await resolveTtsProviderContext('hello', { provider: 'openai' })
    vi.mocked(getOpenAITTSModel).mockResolvedValue('model-b')
    vi.mocked(getOpenAITTSVoice).mockResolvedValue('voice-b')

    await context.synthesize?.('next segment')

    expect(generateOpenAITTS).toHaveBeenCalledWith({
      text: 'next segment',
      model: 'model-a',
      voice: 'voice-a',
      speed: undefined,
      signal: undefined
    })
  })

  it('passes abort signals through OpenAI synthesis', async () => {
    const signal = new AbortController().signal
    vi.mocked(generateOpenAITTS).mockResolvedValueOnce(new ArrayBuffer(8))
    const context = await resolveTtsProviderContext('hello', { provider: 'openai' })

    await context.synthesize?.('hello', { signal })

    expect(generateOpenAITTS).toHaveBeenCalledWith(
      expect.objectContaining({ signal })
    )
  })

  it('passes abort signals through ElevenLabs synthesis', async () => {
    const signal = new AbortController().signal
    vi.mocked(getElevenLabsApiKey).mockResolvedValueOnce('eleven-key')
    vi.mocked(getElevenLabsModel).mockResolvedValueOnce('model-a')
    vi.mocked(getElevenLabsVoiceId).mockResolvedValueOnce('voice-a')
    vi.mocked(generateSpeech).mockResolvedValueOnce(new ArrayBuffer(8))
    const context = await resolveTtsProviderContext('hello', { provider: 'elevenlabs' })

    await context.synthesize?.('hello', { signal })

    expect(generateSpeech).toHaveBeenCalledWith(
      'eleven-key',
      'hello',
      'voice-a',
      'model-a',
      undefined,
      { signal }
    )
  })

  it('uses scoped browser voice listeners without overwriting global handlers', () => {
    const originalHandler = vi.fn()
    let voices: SpeechSynthesisVoice[] = []
    const listeners: EventListener[] = []
    const synthesis = {
      onvoiceschanged: originalHandler,
      getVoices: vi.fn(() => voices),
      addEventListener: vi.fn((_eventName: string, listener: EventListener) => {
        listeners.push(listener)
      }),
      removeEventListener: vi.fn()
    } as unknown as SpeechSynthesis
    const firstUtterance = {} as SpeechSynthesisUtterance
    const secondUtterance = {} as SpeechSynthesisUtterance
    const voice = { name: 'Browser Voice' } as SpeechSynthesisVoice

    const cleanupFirst = applyBrowserSpeechSynthesisVoice(
      firstUtterance,
      synthesis,
      'Browser Voice'
    )
    const cleanupSecond = applyBrowserSpeechSynthesisVoice(
      secondUtterance,
      synthesis,
      'Browser Voice'
    )
    voices = [voice]
    listeners.forEach((listener) => listener(new Event('voiceschanged')))

    expect(synthesis.onvoiceschanged).toBe(originalHandler)
    expect(synthesis.addEventListener).toHaveBeenCalledTimes(2)
    expect(firstUtterance.voice).toBe(voice)
    expect(secondUtterance.voice).toBe(voice)

    cleanupFirst()
    cleanupSecond()

    expect(synthesis.removeEventListener).toHaveBeenCalledTimes(2)
    expect(synthesis.removeEventListener).toHaveBeenCalledWith(
      'voiceschanged',
      listeners[0]
    )
    expect(synthesis.removeEventListener).toHaveBeenCalledWith(
      'voiceschanged',
      listeners[1]
    )
  })

  it('captures reusable text normalization at provider resolution time', async () => {
    vi.mocked(markdownToText).mockImplementation((text: string) => `plain:${text}`)

    const context = await resolveTtsProviderContext('hello', { provider: 'browser' })
    vi.mocked(isSSMLEnabled).mockResolvedValue(true)

    expect(context.utterance).toBe('plain:hello')
    expect(context.normalizeText?.('next')).toBe('plain:next')
  })

  it('captures the configured browser voice name for read-along playback', async () => {
    vi.mocked(getVoice).mockResolvedValueOnce('Voice A')

    const context = await resolveTtsProviderContext('hello', { provider: 'browser' })

    expect(context.browserVoiceName).toBe('Voice A')
  })
})
