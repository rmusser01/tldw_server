import type {
  WorkspaceBannerImage,
  WorkspaceBannerImageMimeType
} from "@/types/workspace"

export const WORKSPACE_BANNER_ALLOWED_MIME_TYPES = new Set<
  WorkspaceBannerImageMimeType
>(["image/jpeg", "image/png", "image/webp"])
export const WORKSPACE_BANNER_MAX_EDGE_PX = 1400
export const WORKSPACE_BANNER_MAX_BYTES = 350 * 1024
const WORKSPACE_BANNER_WEBP_QUALITY = 0.86

export type WorkspaceBannerImageNormalizationErrorCode =
  | "unsupported_mime_type"
  | "image_decode_failed"
  | "image_encode_failed"
  | "image_too_large"

export class WorkspaceBannerImageNormalizationError extends Error {
  code: WorkspaceBannerImageNormalizationErrorCode

  constructor(
    code: WorkspaceBannerImageNormalizationErrorCode,
    message: string
  ) {
    super(message)
    this.name = "WorkspaceBannerImageNormalizationError"
    this.code = code
  }
}

type WorkspaceBannerImageDimensions = {
  width: number
  height: number
}

type WorkspaceBannerImageNormalizerDeps = {
  readFileAsDataUrl?: (file: File) => Promise<string>
  decodeImageDimensions?: (
    dataUrl: string
  ) => Promise<WorkspaceBannerImageDimensions>
  encodeImageToWebp?: (
    dataUrl: string,
    dimensions: WorkspaceBannerImageDimensions,
    quality: number
  ) => Promise<string | null>
  now?: () => Date
}

export type NormalizeWorkspaceBannerImageOptions = {
  maxEdgePx?: number
  maxBytes?: number
  quality?: number
  deps?: WorkspaceBannerImageNormalizerDeps
}

const isAllowedMimeType = (
  mimeType: string
): mimeType is WorkspaceBannerImageMimeType =>
  WORKSPACE_BANNER_ALLOWED_MIME_TYPES.has(
    mimeType as WorkspaceBannerImageMimeType
  )

const normalizeMimeType = (mimeType: string): string =>
  mimeType.trim().toLowerCase()

const parseDataUrlMimeType = (
  dataUrl: string
): WorkspaceBannerImageMimeType | null => {
  const match = /^data:([^;,]+);base64,/i.exec(dataUrl)
  if (!match || typeof match[1] !== "string") return null
  const normalized = normalizeMimeType(match[1])
  return isAllowedMimeType(normalized) ? normalized : null
}

const readFileAsDataUrl = async (file: File): Promise<string> => {
  const blobCandidate = file as File & {
    arrayBuffer?: () => Promise<ArrayBuffer>
  }
  if (typeof blobCandidate.arrayBuffer === "function") {
    const raw = await blobCandidate.arrayBuffer()
    const bytes = new Uint8Array(raw)
    let binary = ""
    bytes.forEach((byte) => {
      binary += String.fromCharCode(byte)
    })
    return `data:${file.type || "application/octet-stream"};base64,${btoa(binary)}`
  }

  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result)
        return
      }
      reject(new Error("Failed to read file"))
    }
    reader.onerror = () => reject(reader.error || new Error("Failed to read file"))
    reader.readAsDataURL(file)
  })
}

const decodeImageDimensions = async (
  dataUrl: string
): Promise<WorkspaceBannerImageDimensions> =>
  new Promise<WorkspaceBannerImageDimensions>((resolve, reject) => {
    const image = new Image()
    image.onload = () =>
      resolve({
        width: image.naturalWidth || image.width,
        height: image.naturalHeight || image.height
      })
    image.onerror = () => reject(new Error("Failed to decode image"))
    image.src = dataUrl
  })

const encodeImageToWebp = async (
  dataUrl: string,
  dimensions: WorkspaceBannerImageDimensions,
  quality: number
): Promise<string | null> => {
  if (typeof document === "undefined") return null
  const canvas = document.createElement("canvas")
  if (!canvas || typeof canvas.getContext !== "function") return null

  canvas.width = dimensions.width
  canvas.height = dimensions.height
  const context = canvas.getContext("2d")
  if (!context) return null

  await new Promise<void>((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      context.drawImage(image, 0, 0, dimensions.width, dimensions.height)
      resolve()
    }
    image.onerror = () => reject(new Error("Failed to decode image for canvas"))
    image.src = dataUrl
  })

  const encoded = canvas.toDataURL("image/webp", quality)
  if (typeof encoded !== "string" || !encoded.startsWith("data:image/webp;base64,")) {
    return null
  }
  return encoded
}

const scaleDimensions = (
  original: WorkspaceBannerImageDimensions,
  maxEdgePx: number
): WorkspaceBannerImageDimensions => {
  const width = Math.max(1, Math.round(original.width))
  const height = Math.max(1, Math.round(original.height))
  const longestEdge = Math.max(width, height)
  if (longestEdge <= maxEdgePx) {
    return { width, height }
  }

  const scale = maxEdgePx / longestEdge
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale))
  }
}

const estimateBase64Bytes = (dataUrl: string): number => {
  const delimiterIndex = dataUrl.indexOf(",")
  if (delimiterIndex < 0) return 0
  const base64 = dataUrl.slice(delimiterIndex + 1)
  if (!base64) return 0

  const padding = base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0
  return Math.max(0, Math.floor((base64.length * 3) / 4) - padding)
}

export async function normalizeWorkspaceBannerImage(
  file: File,
  options: NormalizeWorkspaceBannerImageOptions = {}
): Promise<WorkspaceBannerImage> {
  const normalizedInputMimeType = normalizeMimeType(file.type || "")
  if (!isAllowedMimeType(normalizedInputMimeType)) {
    throw new WorkspaceBannerImageNormalizationError(
      "unsupported_mime_type",
      "Only JPEG, PNG, and WebP banner images are supported."
    )
  }

  const maxEdgePx = options.maxEdgePx ?? WORKSPACE_BANNER_MAX_EDGE_PX
  const maxBytes = options.maxBytes ?? WORKSPACE_BANNER_MAX_BYTES
  const quality = options.quality ?? WORKSPACE_BANNER_WEBP_QUALITY
  const deps = options.deps ?? {}

  const readDataUrl = deps.readFileAsDataUrl || readFileAsDataUrl
  const decodeDimensions = deps.decodeImageDimensions || decodeImageDimensions
  const encodeToWebp = deps.encodeImageToWebp || encodeImageToWebp
  const now = deps.now || (() => new Date())

  const inputDataUrl = await readDataUrl(file)

  let originalDimensions: WorkspaceBannerImageDimensions
  try {
    originalDimensions = await decodeDimensions(inputDataUrl)
  } catch {
    throw new WorkspaceBannerImageNormalizationError(
      "image_decode_failed",
      "Failed to decode banner image."
    )
  }

  const targetDimensions = scaleDimensions(originalDimensions, maxEdgePx)
  let outputDataUrl = inputDataUrl
  let outputMimeType: WorkspaceBannerImageMimeType = normalizedInputMimeType

  try {
    const encoded = await encodeToWebp(inputDataUrl, targetDimensions, quality)
    if (encoded) {
      const parsedMimeType = parseDataUrlMimeType(encoded)
      if (parsedMimeType) {
        outputDataUrl = encoded
        outputMimeType = parsedMimeType
      }
    }
  } catch {
    throw new WorkspaceBannerImageNormalizationError(
      "image_encode_failed",
      "Failed to normalize banner image."
    )
  }

  const normalizedBytes = estimateBase64Bytes(outputDataUrl)
  if (normalizedBytes > maxBytes) {
    throw new WorkspaceBannerImageNormalizationError(
      "image_too_large",
      "Normalized banner image exceeds the maximum supported size."
    )
  }

  return {
    dataUrl: outputDataUrl,
    mimeType: outputMimeType,
    width: targetDimensions.width,
    height: targetDimensions.height,
    bytes: normalizedBytes,
    updatedAt: now()
  }
}
