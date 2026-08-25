import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { PersonaVisualAsset } from "@/types/persona-visuals"

const DEFAULT_MAX_BYTES = 16 * 1024 * 1024
const SHA256_HEX_PATTERN = /^[0-9a-f]{64}$/

type AssetInput = Pick<
  PersonaVisualAsset,
  "id" | "url" | "checksum_sha256" | "byte_size" | "mime_type"
> & {
  checksum_sha256: string
  byte_size: number
}

type AssetEntry = {
  asset: AssetInput
  controller: AbortController
  promise: Promise<void>
  url?: string
  references: number
  waiters: number
  cleared: boolean
  revoked: boolean
}

export type PersonaVisualAssetHandle = {
  readonly url: string
  readonly mimeType: string
  release(): void
}

export class PersonaVisualAssetError extends Error {
  constructor(readonly code: string) {
    super(code)
    this.name = "PersonaVisualAssetError"
  }
}

const assetCache = new Map<string, AssetEntry>()

const cacheKey = (asset: AssetInput): string =>
  `${asset.id}:${asset.checksum_sha256}`

const fail = (code: string): never => {
  throw new PersonaVisualAssetError(code)
}

const assertDeclaredAsset = (asset: AssetInput, maxBytes: number): void => {
  if (!Number.isSafeInteger(asset.byte_size) || asset.byte_size < 0) {
    fail("asset_size_mismatch")
  }
  if (asset.byte_size > maxBytes) fail("asset_too_large")
  if (!SHA256_HEX_PATTERN.test(asset.checksum_sha256)) {
    fail("asset_checksum_mismatch")
  }
  if (!Object.hasOwn(SIGNATURES, asset.mime_type)) fail("asset_mime_mismatch")
}

const SIGNATURES: Record<string, number[]> = {
  "image/png": [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
  "image/jpeg": [0xff, 0xd8, 0xff],
  "image/gif": [0x47, 0x49, 0x46],
  "image/webp": [0x52, 0x49, 0x46, 0x46]
}

const signatureMatches = (mimeType: string, bytes: Uint8Array): boolean => {
  const signature = SIGNATURES[mimeType]
  if (!signature || bytes.byteLength < signature.length) return false
  if (!signature.every((value, index) => bytes[index] === value)) return false
  if (mimeType === "image/gif") {
    return bytes.byteLength >= 6
      && bytes[3] === 0x38
      && (bytes[4] === 0x37 || bytes[4] === 0x39)
      && bytes[5] === 0x61
  }
  return mimeType !== "image/webp"
    || bytes.byteLength >= 12
      && bytes[8] === 0x57
      && bytes[9] === 0x45
      && bytes[10] === 0x42
      && bytes[11] === 0x50
}

const sha256 = async (bytes: ArrayBuffer): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")
}

const revoke = (entry: AssetEntry): void => {
  if (!entry.url || entry.revoked) return
  URL.revokeObjectURL(entry.url)
  entry.revoked = true
}

const remove = (entry: AssetEntry): void => {
  const key = cacheKey(entry.asset)
  if (assetCache.get(key) === entry) assetCache.delete(key)
}

const discardUnretained = (entry: AssetEntry): void => {
  if (entry.references || entry.waiters) return
  if (entry.url) {
    remove(entry)
    revoke(entry)
    return
  }
  remove(entry)
  entry.controller.abort()
}

const load = async (entry: AssetEntry, maxBytes: number): Promise<void> => {
  try {
    const response = await tldwClient.fetchWithAuth(entry.asset.url as any, {
      responseType: "arrayBuffer",
      signal: entry.controller.signal
    })
    if (!response.ok) fail("asset_fetch_failed")
    if (!(response.data instanceof ArrayBuffer)) fail("asset_response_malformed")

    const bytes = response.data
    if (bytes.byteLength > maxBytes) fail("asset_too_large")
    if (bytes.byteLength !== entry.asset.byte_size) fail("asset_size_mismatch")
    if (!signatureMatches(entry.asset.mime_type, new Uint8Array(bytes))) {
      fail("asset_mime_mismatch")
    }
    if ((await sha256(bytes)) !== entry.asset.checksum_sha256) {
      fail("asset_checksum_mismatch")
    }
    if (entry.controller.signal.aborted || entry.cleared) fail("asset_aborted")

    entry.url = URL.createObjectURL(new Blob([bytes], { type: entry.asset.mime_type }))
    if (entry.cleared) {
      revoke(entry)
      fail("asset_aborted")
    }
  } catch (error) {
    remove(entry)
    if (entry.controller.signal.aborted || entry.cleared) fail("asset_aborted")
    throw error
  }
}

const waitFor = async (
  entry: AssetEntry,
  signal: AbortSignal | undefined,
  releaseWaiter: () => void
): Promise<void> => {
  if (signal?.aborted) fail("asset_aborted")
  if (!signal) return entry.promise
  await Promise.race([
    entry.promise,
    new Promise<never>((_, reject) => {
      signal.addEventListener(
        "abort",
        () => {
          releaseWaiter()
          reject(new PersonaVisualAssetError("asset_aborted"))
        },
        { once: true }
      )
    })
  ])
}

export async function acquirePersonaVisualAsset(
  asset: AssetInput,
  options: { signal?: AbortSignal; maxBytes?: number } = {}
): Promise<PersonaVisualAssetHandle> {
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) fail("asset_too_large")
  assertDeclaredAsset(asset, maxBytes)
  if (options.signal?.aborted) fail("asset_aborted")

  const key = cacheKey(asset)
  let entry = assetCache.get(key)
  if (!entry) {
    const controller = new AbortController()
    entry = {
      asset,
      controller,
      promise: Promise.resolve(),
      references: 0,
      waiters: 0,
      cleared: false,
      revoked: false
    }
    assetCache.set(key, entry)
    entry.promise = load(entry, maxBytes)
  } else if (
    entry.asset.byte_size !== asset.byte_size
    || entry.asset.mime_type !== asset.mime_type
  ) {
    fail("asset_size_mismatch")
  }

  entry.waiters += 1
  let waiting = true
  const releaseWaiter = () => {
    if (!waiting) return
    waiting = false
    entry.waiters -= 1
    discardUnretained(entry)
  }
  try {
    await waitFor(entry, options.signal, releaseWaiter)
    if (!entry.url || entry.cleared || entry.revoked) fail("asset_aborted")
    entry.references += 1
    let released = false
    return {
      url: entry.url,
      mimeType: entry.asset.mime_type,
      release: () => {
        if (released) return
        released = true
        entry.references -= 1
        discardUnretained(entry)
      }
    }
  } finally {
    releaseWaiter()
  }
}

export const clearPersonaVisualAssetCache = (): void => {
  for (const entry of assetCache.values()) {
    entry.cleared = true
    entry.controller.abort()
    revoke(entry)
  }
  assetCache.clear()
}
