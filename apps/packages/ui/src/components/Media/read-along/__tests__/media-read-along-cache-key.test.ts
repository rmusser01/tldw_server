import { describe, expect, it } from 'vitest'

import {
  buildReadAlongCacheKey,
  buildTtsSettingsSignature,
  sha256Hex
} from '../media-read-along-cache-key'

describe('media read-along cache keys', () => {
  it('does not include raw segment text in the stable key metadata', async () => {
    const key = await buildReadAlongCacheKey({
      serverScope: 'http://127.0.0.1:8000',
      mediaId: '42',
      mediaKind: 'media',
      segmentId: 's1',
      segmentText: 'private transcript text',
      sourceStart: 10,
      sourceEnd: 33,
      settingsSignature: 'provider:tldw|voice:default'
    })

    expect(key.id).not.toContain('private transcript text')
    expect(JSON.stringify(key)).not.toContain('private transcript text')
  })

  it('hashes segment text with a lowercase SHA-256 hex digest', async () => {
    const key = await buildReadAlongCacheKey({
      serverScope: 'http://127.0.0.1:8000',
      mediaId: '42',
      mediaKind: 'media',
      segmentId: 's1',
      segmentText: 'private transcript text',
      sourceStart: 10,
      sourceEnd: 33,
      settingsSignature: 'provider:tldw|voice:default'
    })

    expect(key.textHash).toMatch(/^[a-f0-9]{64}$/)
  })

  it('matches the known SHA-256 digest for abc', async () => {
    await expect(sha256Hex('abc')).resolves.toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    )
  })

  it('fails closed instead of producing a fake SHA-256 digest when Web Crypto is unavailable', async () => {
    const originalCrypto = globalThis.crypto
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: undefined
    })

    try {
      await expect(sha256Hex('abc')).rejects.toThrow(/SHA-256/)
    } finally {
      Object.defineProperty(globalThis, 'crypto', {
        configurable: true,
        value: originalCrypto
      })
    }
  })

  it('changes settings signature when voice or speed changes', () => {
    const base = buildTtsSettingsSignature({
      provider: 'tldw',
      model: 'kokoro',
      voice: 'af',
      speed: 1,
      format: 'mp3'
    })
    const changedVoice = buildTtsSettingsSignature({
      provider: 'tldw',
      model: 'kokoro',
      voice: 'bf',
      speed: 1,
      format: 'mp3'
    })
    const changedSpeed = buildTtsSettingsSignature({
      provider: 'tldw',
      model: 'kokoro',
      voice: 'af',
      speed: 1.2,
      format: 'mp3'
    })

    expect(base).not.toEqual(changedVoice)
    expect(base).not.toEqual(changedSpeed)
  })

  it('preserves opaque model and voice id casing in settings signatures', () => {
    const mixedCase = buildTtsSettingsSignature({
      provider: ' OpenAI ',
      model: ' TTS-Model-A ',
      voice: ' Voice-ID-A ',
      speed: 1,
      format: ' MP3 ',
      language: ' EN-US '
    })
    const lowerCaseIds = buildTtsSettingsSignature({
      provider: 'openai',
      model: 'tts-model-a',
      voice: 'voice-id-a',
      speed: 1,
      format: 'mp3',
      language: 'en-us'
    })

    expect(mixedCase).toContain('provider:openai')
    expect(mixedCase).toContain('model:TTS-Model-A')
    expect(mixedCase).toContain('voice:Voice-ID-A')
    expect(mixedCase).toContain('format:mp3')
    expect(mixedCase).toContain('language:en-us')
    expect(mixedCase).not.toEqual(lowerCaseIds)
  })
})
