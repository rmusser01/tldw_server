/**
 * Character Export Utilities
 * Export characters to JSON or PNG (with embedded metadata) format
 */

/**
 * Character V3 data type
 * Based on the SillyTavern Character Card v3 specification
 */
export interface CharacterV3 {
  name?: string
  description?: string
  personality?: string
  scenario?: string
  system_prompt?: string
  first_message?: string
  message_example?: string
  creator_notes?: string
  tags?: string[]
  alternate_greetings?: string[]
  creator?: string
  character_version?: string
  extensions?: Record<string, unknown>
  post_history_instructions?: string
  avatar_url?: string
  image_base64?: string
  [key: string]: unknown
}

/**
 * V3 Character Card format with embedded data
 * Based on the SillyTavern Character Card v3 specification
 */
export interface CharacterCardV3 {
  spec: "chara_card_v3"
  spec_version: "3.0"
  data: CharacterV3
}

/**
 * Sanitize a filename for safe download
 */
export function sanitizeFilename(name: string): string {
  return name
    .replace(/[/\\?%*:|"<>]/g, "-")
    .replace(/\s+/g, "_")
    .substring(0, 100)
}

/**
 * Trigger a file download in the browser
 */
export function downloadFile(content: string | Blob, filename: string, mimeType?: string) {
  const blob = content instanceof Blob
    ? content
    : new Blob([content], { type: mimeType || "application/octet-stream" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Export character data to JSON file
 */
export function exportCharacterToJSON(
  character: CharacterV3,
  filename?: string
): void {
  const name = character.name || "character"
  const safeFilename = filename || `${sanitizeFilename(name)}_character.json`

  const cardData: CharacterCardV3 = {
    spec: "chara_card_v3",
    spec_version: "3.0",
    data: character
  }

  const json = JSON.stringify(cardData, null, 2)
  downloadFile(json, safeFilename, "application/json")
}

/**
 * Export multiple characters to a single JSON file
 */
export function exportCharactersToJSON(
  characters: CharacterV3[],
  filename?: string
): void {
  const safeFilename = filename || `characters_export_${Date.now()}.json`

  const cardData = characters.map((character) => ({
    spec: "chara_card_v3" as const,
    spec_version: "3.0" as const,
    data: character
  }))

  const json = JSON.stringify(cardData, null, 2)
  downloadFile(json, safeFilename, "application/json")
}

/**
 * Embed character metadata into a PNG image using tEXt chunk
 *
 * PNG files support text metadata through tEXt, zTXt (compressed), or iTXt (international) chunks.
 * Character card data is stored in a tEXt chunk with keyword "chara" and base64-encoded JSON value.
 *
 * This follows the SillyTavern/TavernAI character card PNG format specification.
 */
export async function embedMetadataInPNG(
  imageData: ArrayBuffer | Blob,
  character: CharacterV3
): Promise<Blob> {
  // Convert Blob to ArrayBuffer if needed
  const buffer = imageData instanceof Blob
    ? await imageData.arrayBuffer()
    : imageData

  const uint8 = new Uint8Array(buffer)

  // Validate PNG signature (first 8 bytes)
  const pngSignature = [137, 80, 78, 71, 13, 10, 26, 10]
  for (let i = 0; i < 8; i++) {
    if (uint8[i] !== pngSignature[i]) {
      throw new Error("Invalid PNG file: signature mismatch")
    }
  }

  // Create the character card data
  const cardData: CharacterCardV3 = {
    spec: "chara_card_v3",
    spec_version: "3.0",
    data: character
  }

  // Encode character data as base64
  const jsonString = JSON.stringify(cardData)
  const base64Data = btoa(unescape(encodeURIComponent(jsonString)))

  // Create tEXt chunk with "chara" keyword
  const textChunk = createPNGTextChunk("chara", base64Data)

  // Find position to insert the tEXt chunk (after IHDR, before IDAT)
  // We'll insert right after the IHDR chunk for simplicity
  let insertPosition = 8 // After PNG signature

  // Skip IHDR chunk (it's always first after signature)
  // Chunk structure: 4 bytes length + 4 bytes type + data + 4 bytes CRC
  const ihdrLength = (uint8[8] << 24) | (uint8[9] << 16) | (uint8[10] << 8) | uint8[11]
  insertPosition = 8 + 4 + 4 + ihdrLength + 4 // signature + length + type + data + crc

  // Build new PNG with embedded metadata
  const result = new Uint8Array(uint8.length + textChunk.length)
  result.set(uint8.subarray(0, insertPosition), 0)
  result.set(textChunk, insertPosition)
  result.set(uint8.subarray(insertPosition), insertPosition + textChunk.length)

  return new Blob([result], { type: "image/png" })
}

/**
 * Create a PNG tEXt chunk
 * Format: length (4 bytes) + "tEXt" (4 bytes) + keyword + null + text + CRC (4 bytes)
 */
function createPNGTextChunk(keyword: string, text: string): Uint8Array {
  const keywordBytes = new TextEncoder().encode(keyword)
  const textBytes = new TextEncoder().encode(text)

  // Data = keyword + null separator + text
  const dataLength = keywordBytes.length + 1 + textBytes.length

  // Total chunk size = 4 (length) + 4 (type) + data + 4 (CRC)
  const chunk = new Uint8Array(4 + 4 + dataLength + 4)

  // Length (big-endian)
  chunk[0] = (dataLength >> 24) & 0xff
  chunk[1] = (dataLength >> 16) & 0xff
  chunk[2] = (dataLength >> 8) & 0xff
  chunk[3] = dataLength & 0xff

  // Type: "tEXt"
  chunk[4] = 0x74 // 't'
  chunk[5] = 0x45 // 'E'
  chunk[6] = 0x58 // 'X'
  chunk[7] = 0x74 // 't'

  // Data: keyword + null + text
  chunk.set(keywordBytes, 8)
  chunk[8 + keywordBytes.length] = 0 // null separator
  chunk.set(textBytes, 8 + keywordBytes.length + 1)

  // Calculate CRC32 over type + data
  const crcData = chunk.subarray(4, 4 + 4 + dataLength)
  const crc = calculateCRC32(crcData)

  // Write CRC (big-endian)
  const crcOffset = 4 + 4 + dataLength
  chunk[crcOffset] = (crc >> 24) & 0xff
  chunk[crcOffset + 1] = (crc >> 16) & 0xff
  chunk[crcOffset + 2] = (crc >> 8) & 0xff
  chunk[crcOffset + 3] = crc & 0xff

  return chunk
}

/**
 * CRC32 calculation for PNG chunks
 * Uses the standard PNG CRC polynomial
 */
function calculateCRC32(data: Uint8Array): number {
  // CRC table (lazily initialized)
  if (!crc32Table) {
    crc32Table = new Uint32Array(256)
    for (let n = 0; n < 256; n++) {
      let c = n
      for (let k = 0; k < 8; k++) {
        if (c & 1) {
          c = 0xedb88320 ^ (c >>> 1)
        } else {
          c = c >>> 1
        }
      }
      crc32Table[n] = c
    }
  }

  let crc = 0xffffffff
  for (let i = 0; i < data.length; i++) {
    crc = crc32Table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}

let crc32Table: Uint32Array | null = null

/** Maximum bytes to accept when fetching a remote avatar for PNG embedding. */
const MAX_AVATAR_FETCH_BYTES = 5 * 1024 * 1024
/** Timeout for the (allowlisted) remote avatar fetch. */
const AVATAR_FETCH_TIMEOUT_MS = 10_000

/**
 * Decode a base64 (optionally `data:`-prefixed) string into an ArrayBuffer.
 */
function base64ToArrayBuffer(value: string): ArrayBuffer {
  const base64 = value.replace(/^data:image\/\w+;base64,/, "")
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes.buffer
}

/**
 * Decide whether an avatar URL is safe to fetch for PNG export.
 *
 * A character card's `avatar_url` is attacker-controllable (via shared/imported
 * cards), so blindly fetching it would let a card fire an outbound request from
 * the victim's browser (tracking beacon / internal-network probe). We therefore
 * only permit same-origin URLs plus any explicitly-allowed origin (e.g. the
 * configured tldw server). `data:` URLs are handled separately by the caller
 * (decoded locally, no network request).
 */
export function isSafeAvatarFetchUrl(
  rawUrl: string,
  allowedOrigins?: string[]
): boolean {
  if (typeof rawUrl !== "string" || rawUrl.trim().length === 0) return false

  let parsed: URL
  try {
    const base =
      typeof window !== "undefined" && window.location
        ? window.location.href
        : undefined
    parsed = new URL(rawUrl, base)
  } catch {
    return false
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false

  const allowed = new Set<string>()
  if (typeof window !== "undefined" && window.location?.origin) {
    allowed.add(window.location.origin)
  }
  for (const origin of allowedOrigins ?? []) {
    if (typeof origin !== "string" || !origin.trim()) continue
    try {
      allowed.add(new URL(origin).origin)
    } catch {
      // ignore malformed allowlist entries
    }
  }

  return allowed.has(parsed.origin)
}

/**
 * Load avatar image bytes for PNG embedding, defending against SSRF / beacon
 * abuse. Returns null (never throws) when the avatar cannot be loaded safely;
 * callers should fall back to a locally-generated placeholder.
 */
async function loadAvatarImageData(
  avatarUrl: string,
  allowedOrigins?: string[]
): Promise<ArrayBuffer | null> {
  // Inline data: URLs never hit the network.
  if (avatarUrl.startsWith("data:")) {
    try {
      return base64ToArrayBuffer(avatarUrl)
    } catch {
      console.warn("Character export: failed to decode inline avatar data URL")
      return null
    }
  }

  if (!isSafeAvatarFetchUrl(avatarUrl, allowedOrigins)) {
    console.warn(
      "Character export: skipping avatar fetch for a non-allowlisted URL; exporting without the embedded avatar"
    )
    return null
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), AVATAR_FETCH_TIMEOUT_MS)
  try {
    const response = await fetch(avatarUrl, {
      signal: controller.signal,
      // Do not attach the user's cookies/credentials to the avatar request.
      credentials: "omit"
    })
    if (!response.ok) {
      console.warn(
        `Character export: avatar fetch failed (${response.status}); exporting without the embedded avatar`
      )
      return null
    }
    const declaredLength = Number(response.headers.get("content-length"))
    if (
      Number.isFinite(declaredLength) &&
      declaredLength > MAX_AVATAR_FETCH_BYTES
    ) {
      console.warn(
        "Character export: avatar exceeds the size cap; exporting without the embedded avatar"
      )
      return null
    }
    const buffer = await response.arrayBuffer()
    if (buffer.byteLength > MAX_AVATAR_FETCH_BYTES) {
      console.warn(
        "Character export: avatar exceeds the size cap; exporting without the embedded avatar"
      )
      return null
    }
    return buffer
  } catch (error) {
    console.warn("Character export: avatar fetch aborted or failed", error)
    return null
  } finally {
    clearTimeout(timeout)
  }
}

/**
 * Export character to PNG with embedded metadata
 *
 * If the character has an already-local avatar (base64), it is embedded directly.
 * A remote `avatarUrl` is only fetched when it is same-origin or from an allowed
 * origin (see `isSafeAvatarFetchUrl`), with a timeout and response-size cap.
 * Otherwise a locally-generated placeholder image is used.
 */
export async function exportCharacterToPNG(
  character: CharacterV3,
  options?: {
    avatarUrl?: string
    avatarBase64?: string
    filename?: string
    /**
     * Extra origins (besides the WebUI origin) whose avatars may be fetched,
     * e.g. the configured tldw server origin.
     */
    allowedAvatarOrigins?: string[]
  }
): Promise<void> {
  const name = character.name || "character"
  const filename = options?.filename || `${sanitizeFilename(name)}_character.png`

  let imageData: ArrayBuffer | null = null

  // Prefer already-local base64 (no network request).
  if (options?.avatarBase64) {
    try {
      imageData = base64ToArrayBuffer(options.avatarBase64)
    } catch {
      console.warn("Character export: failed to decode provided avatar base64")
      imageData = null
    }
  } else if (options?.avatarUrl) {
    imageData = await loadAvatarImageData(
      options.avatarUrl,
      options.allowedAvatarOrigins
    )
  }

  if (!imageData) {
    // Fall back to a locally-generated placeholder image.
    imageData = await createPlaceholderPNG(name)
  }

  // Embed character metadata into PNG
  const pngWithMetadata = await embedMetadataInPNG(imageData, character)

  // Download the file
  downloadFile(pngWithMetadata, filename, "image/png")
}

/**
 * Create a simple placeholder PNG image for characters without avatars
 * Creates a 256x256 image with the first letter of the character's name
 */
async function createPlaceholderPNG(name: string): Promise<ArrayBuffer> {
  // Create canvas
  const canvas = document.createElement("canvas")
  canvas.width = 256
  canvas.height = 256
  const ctx = canvas.getContext("2d")

  if (!ctx) {
    throw new Error("Failed to get canvas context")
  }

  // Generate a consistent color based on the name
  const hash = name.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0)
  const hue = hash % 360

  // Draw background gradient
  const gradient = ctx.createLinearGradient(0, 0, 256, 256)
  gradient.addColorStop(0, `hsl(${hue}, 60%, 50%)`)
  gradient.addColorStop(1, `hsl(${(hue + 30) % 360}, 60%, 40%)`)
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 256, 256)

  // Draw first letter
  const letter = (name[0] || "?").toUpperCase()
  ctx.fillStyle = "white"
  ctx.font = "bold 120px -apple-system, BlinkMacSystemFont, sans-serif"
  ctx.textAlign = "center"
  ctx.textBaseline = "middle"
  ctx.fillText(letter, 128, 128)

  // Convert to PNG blob and then ArrayBuffer
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        blob.arrayBuffer().then(resolve).catch(reject)
      } else {
        reject(new Error("Failed to create placeholder image"))
      }
    }, "image/png")
  })
}

/**
 * Read character metadata from a PNG file
 * Returns null if no character metadata is found
 */
export async function readCharacterFromPNG(
  imageData: ArrayBuffer | Blob
): Promise<CharacterV3 | null> {
  // Convert Blob to ArrayBuffer if needed
  const buffer = imageData instanceof Blob
    ? await imageData.arrayBuffer()
    : imageData

  const uint8 = new Uint8Array(buffer)

  // Validate PNG signature
  const pngSignature = [137, 80, 78, 71, 13, 10, 26, 10]
  for (let i = 0; i < 8; i++) {
    if (uint8[i] !== pngSignature[i]) {
      return null // Not a valid PNG
    }
  }

  // Parse PNG chunks looking for tEXt chunk with "chara" keyword
  let offset = 8 // Skip PNG signature

  while (offset < uint8.length) {
    // Read chunk length (big-endian)
    const length = (uint8[offset] << 24) | (uint8[offset + 1] << 16) |
                   (uint8[offset + 2] << 8) | uint8[offset + 3]

    // Read chunk type
    const type = String.fromCharCode(
      uint8[offset + 4],
      uint8[offset + 5],
      uint8[offset + 6],
      uint8[offset + 7]
    )

    if (type === "IEND") {
      break // End of PNG
    }

    if (type === "tEXt") {
      // Read chunk data
      const dataStart = offset + 8
      const data = uint8.subarray(dataStart, dataStart + length)

      // Find null separator between keyword and text
      const nullIndex = data.indexOf(0)
      if (nullIndex > 0) {
        const keyword = new TextDecoder().decode(data.subarray(0, nullIndex))

        if (keyword === "chara") {
          // Found character data
          const textData = new TextDecoder().decode(data.subarray(nullIndex + 1))

          try {
            // Decode base64
            const jsonString = decodeURIComponent(escape(atob(textData)))
            const cardData = JSON.parse(jsonString) as CharacterCardV3

            if (cardData.spec === "chara_card_v3" && cardData.data) {
              return cardData.data
            }
          } catch {
            // Invalid data, continue searching
          }
        }
      }
    }

    // Move to next chunk: length (4) + type (4) + data (length) + CRC (4)
    offset += 4 + 4 + length + 4
  }

  return null
}
