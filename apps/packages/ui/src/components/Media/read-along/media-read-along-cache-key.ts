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

export const sha256Hex = async (text: string): Promise<string> => {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) {
    throw new Error('Web Crypto SHA-256 is required for read-along audio cache keys')
  }

  const digest = await subtle.digest('SHA-256', textEncoder.encode(text))
  return toHex(new Uint8Array(digest))
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
