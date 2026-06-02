export const ALLOWED_IMAGE_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp"
])

export const IMAGE_ATTACHMENT_MIME_BY_EXTENSION = new Map<string, string>([
  ["avif", "image/avif"],
  ["bmp", "image/bmp"],
  ["gif", "image/gif"],
  ["heic", "image/heic"],
  ["heif", "image/heif"],
  ["ico", "image/ico"],
  ["jpeg", "image/jpeg"],
  ["jpg", "image/jpeg"],
  ["png", "image/png"],
  ["svg", "image/svg+xml"],
  ["tif", "image/tiff"],
  ["tiff", "image/tiff"],
  ["webp", "image/webp"]
])

export const IMAGE_ATTACHMENT_MIME_TYPES = new Set<string>([
  ...IMAGE_ATTACHMENT_MIME_BY_EXTENSION.values(),
  "image/ico",
  "image/jpg"
])

export const GENERIC_IMAGE_FALLBACK_MIME_TYPES = new Set([
  "",
  "application/octet-stream"
])

export interface ImageAttachmentFileLike {
  name: string
  type?: string | null
}

function getFileExtension(fileName: string): string | null {
  const dotIndex = fileName.lastIndexOf(".")
  if (dotIndex <= 0 || dotIndex === fileName.length - 1) return null

  return fileName.slice(dotIndex + 1).toLowerCase()
}

export function normalizeAttachmentMimeType(
  mimeType: string | null | undefined
): string {
  return typeof mimeType === "string" ? mimeType.trim().toLowerCase() : ""
}

export function inferImageAttachmentMimeType(
  file: ImageAttachmentFileLike
): string | null {
  const fileType = normalizeAttachmentMimeType(file.type)
  if (fileType.startsWith("image/")) return fileType
  if (!GENERIC_IMAGE_FALLBACK_MIME_TYPES.has(fileType)) return null

  const extension = getFileExtension(file.name)
  return extension ? IMAGE_ATTACHMENT_MIME_BY_EXTENSION.get(extension) ?? null : null
}

export function normalizeImageDataUrlMime(
  dataUrl: string,
  mimeType: string | null | undefined
): string {
  if (dataUrl.toLowerCase().startsWith("data:image/")) return dataUrl

  const commaIndex = dataUrl.indexOf(",")
  if (commaIndex === -1) return dataUrl
  if (!mimeType) return dataUrl

  return `data:${mimeType};base64,${dataUrl.slice(commaIndex + 1)}`
}

function isBase64ImageChar(code: number): boolean {
  return (
    (code >= 0x41 && code <= 0x5a) || // A-Z
    (code >= 0x61 && code <= 0x7a) || // a-z
    (code >= 0x30 && code <= 0x39) || // 0-9
    code === 0x2b || // +
    code === 0x2f || // /
    code === 0x5f || // _
    code === 0x2d // -
  )
}

function isValidBase64ImagePayload(value: string): boolean {
  const length = value.length
  if (!length || length % 4 !== 0) return false

  let paddingStart = length
  for (let i = 0; i < length; i += 1) {
    const code = value.charCodeAt(i)
    if (code === 0x3d) {
      paddingStart = i
      break
    }
    if (!isBase64ImageChar(code)) return false
  }

  const paddingLength = length - paddingStart
  if (paddingLength > 2) return false

  for (let i = paddingStart; i < length; i += 1) {
    if (value.charCodeAt(i) !== 0x3d) return false
  }

  return true
}

export function decodeBase64Header(
  value: string,
  maxChars = 128,
  maxBytes = 32
): Uint8Array | null {
  if (typeof atob !== "function") return null
  if (!value) return null

  try {
    const trimmed = value.trim()
    const decoded = atob(trimmed.slice(0, Math.min(trimmed.length, maxChars)))
    const headerBytes = new Uint8Array(Math.min(decoded.length, maxBytes))
    for (let i = 0; i < headerBytes.length; i += 1) {
      headerBytes[i] = decoded.charCodeAt(i)
    }
    return headerBytes
  } catch {
    return null
  }
}

export function detectImageMime(bytes: Uint8Array): string | null {
  const isPng =
    bytes.length >= 4 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  if (isPng) return "image/png"

  const isJpeg =
    bytes.length >= 3 &&
    bytes[0] === 0xff &&
    bytes[1] === 0xd8 &&
    bytes[2] === 0xff
  if (isJpeg) return "image/jpeg"

  const isGif =
    bytes.length >= 6 &&
    bytes[0] === 0x47 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x38 &&
    (bytes[4] === 0x39 || bytes[4] === 0x37) &&
    bytes[5] === 0x61
  if (isGif) return "image/gif"

  const isWebp =
    bytes.length >= 12 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  if (isWebp) return "image/webp"

  return null
}

export function createImageDataUrl(base64: string): string | null {
  if (!base64 || typeof base64 !== "string") return null

  const trimmed = base64.trim()
  if (!trimmed) return null
  if (trimmed.toLowerCase().startsWith("data:image/")) return trimmed
  if (!isValidBase64ImagePayload(trimmed)) return null

  const headerBytes = decodeBase64Header(trimmed)
  if (!headerBytes) return null

  const mime = detectImageMime(headerBytes)
  if (!mime || !ALLOWED_IMAGE_MIME_TYPES.has(mime)) return null

  return `data:${mime};base64,${trimmed}`
}

export function validateAndCreateImageDataUrl(value: unknown): string {
  if (typeof value !== "string") return ""

  const trimmed = value.trim()
  if (!trimmed || trimmed.toLowerCase().startsWith("data:")) return ""

  return createImageDataUrl(trimmed) || ""
}
