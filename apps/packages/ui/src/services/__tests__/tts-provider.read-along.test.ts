import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/services/tts', () => ({
  getOpenAITTSModel: vi.fn(async () => 'tts-1'),
  getOpenAITTSVoice: vi.fn(async () => 'alloy'),
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
import { generateOpenAITTS } from '@/services/openai-tts'
import { getOpenAITTSModel, getOpenAITTSVoice } from '@/services/tts'
import { resolveTtsProviderContext } from '../tts-provider'

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
})
