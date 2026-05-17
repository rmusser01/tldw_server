import { describe, expect, it } from 'vitest'

import {
  buildReadAlongCacheKey,
  buildTtsSettingsSignature
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
})
