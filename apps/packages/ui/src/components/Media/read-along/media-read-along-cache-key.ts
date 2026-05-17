export type ReadAlongCacheKeyInput = {
  serverScope: string
  mediaId: string
  mediaKind: string
  segmentId: string
  segmentText: string
  sourceStart: number
  sourceEnd: number
  settingsSignature: string
}

export type ReadAlongCacheKey = {
  id: string
  mediaId: string
  mediaKind: string
  segmentId: string
  settingsSignature: string
  textHash: string
}

export type TtsSettingsSignatureInput = {
  provider: string
  model?: string | null
  voice?: string | null
  speed?: number | null
  format?: string | null
  language?: string | null
}

const textEncoder = new TextEncoder()

const toHex = (bytes: Uint8Array): string =>
  Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('')

const fallbackHash64 = (text: string): string => {
  let first = 0x811c9dc5
  let second = 0x01000193

  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    first ^= code
    first = Math.imul(first, 0x01000193) >>> 0
    second ^= code + index
    second = Math.imul(second, 0x85ebca6b) >>> 0
  }

  const chunks = [
    first,
    second,
    Math.imul(first ^ second, 0xc2b2ae35) >>> 0,
    Math.imul(first + second, 0x27d4eb2f) >>> 0,
    Math.imul(first ^ text.length, 0x165667b1) >>> 0,
    Math.imul(second ^ text.length, 0xd3a2646c) >>> 0,
    Math.imul(first + text.length, 0x9e3779b1) >>> 0,
    Math.imul(second + text.length, 0x85ebca77) >>> 0
  ]

  return chunks.map((chunk) => chunk.toString(16).padStart(8, '0')).join('')
}

export const sha256Hex = async (text: string): Promise<string> => {
  const subtle = globalThis.crypto?.subtle
  if (subtle) {
    const digest = await subtle.digest('SHA-256', textEncoder.encode(text))
    return toHex(new Uint8Array(digest))
  }

  return fallbackHash64(text)
}

const normalizeSignaturePart = (value: unknown): string =>
  value == null ? '' : String(value).trim().toLowerCase()

export const buildTtsSettingsSignature = ({
  provider,
  model,
  voice,
  speed,
  format,
  language
}: TtsSettingsSignatureInput): string => {
  const normalizedSpeed =
    typeof speed === 'number' && Number.isFinite(speed) ? String(speed) : ''

  return [
    `provider:${normalizeSignaturePart(provider)}`,
    `model:${normalizeSignaturePart(model)}`,
    `voice:${normalizeSignaturePart(voice)}`,
    `speed:${normalizedSpeed}`,
    `format:${normalizeSignaturePart(format)}`,
    `language:${normalizeSignaturePart(language)}`
  ].join('|')
}

export const buildReadAlongCacheKey = async ({
  serverScope,
  mediaId,
  mediaKind,
  segmentId,
  segmentText,
  sourceStart,
  sourceEnd,
  settingsSignature
}: ReadAlongCacheKeyInput): Promise<ReadAlongCacheKey> => {
  const textHash = await sha256Hex(segmentText)
  const stableScope = await sha256Hex(
    [
      serverScope,
      mediaId,
      mediaKind,
      segmentId,
      sourceStart,
      sourceEnd,
      settingsSignature,
      textHash
    ].join('\n')
  )

  return {
    id: `read-along:${stableScope}`,
    mediaId,
    mediaKind,
    segmentId,
    settingsSignature,
    textHash
  }
}
