import { bgRequest, bgUpload } from "@/services/background-proxy"
import type { FlashcardsRequestOptions } from "./flashcards"
import { requestScopeFields } from "./tldw/domains/service-prompts"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export const FLASHCARD_ASSET_SCHEME = "flashcard-asset://"

export interface FlashcardAssetMetadata {
  asset_uuid: string
  reference: string
  markdown_snippet: string
  mime_type: string
  byte_size: number
  width?: number | null
  height?: number | null
  original_filename?: string | null
}

type ObjectUrlCacheEntry = {
  refCount: number
  objectUrl?: string
  promise?: Promise<string>
}

const objectUrlCache = new Map<string, ObjectUrlCacheEntry>()

export const isFlashcardAssetReference = (value: string | null | undefined): boolean =>
  typeof value === "string" && value.startsWith(FLASHCARD_ASSET_SCHEME)

export const parseFlashcardAssetReference = (
  value: string | null | undefined
): string | null => {
  if (!isFlashcardAssetReference(value)) return null
  const assetUuid = String(value).slice(FLASHCARD_ASSET_SCHEME.length).trim()
  return assetUuid || null
}

export async function uploadFlashcardAsset(
  file: File,
  options?: FlashcardsRequestOptions
): Promise<FlashcardAssetMetadata> {
  options?.signal?.throwIfAborted()
  const data = new Uint8Array(await file.arrayBuffer())
  options?.signal?.throwIfAborted()
  return await bgUpload<FlashcardAssetMetadata>({
    path: "/api/v1/flashcards/assets" as AllowedPath,
    method: "POST",
    ...requestScopeFields(options?.requestScope),
    abortSignal: options?.signal,
    fileFieldName: "file",
    file: {
      name: file.name,
      type: file.type,
      data
    }
  })
}

export async function acquireFlashcardAssetObjectUrl(
  assetUuid: string
): Promise<string> {
  const normalizedUuid = String(assetUuid || "").trim()
  if (!normalizedUuid) {
    throw new Error("Flashcard asset UUID is required.")
  }

  const existing = objectUrlCache.get(normalizedUuid)
  if (existing) {
    existing.refCount += 1
    if (existing.objectUrl) return existing.objectUrl
    if (existing.promise) return await existing.promise
  }

  const entry: ObjectUrlCacheEntry = {
    refCount: 1
  }
  objectUrlCache.set(normalizedUuid, entry)

  entry.promise = bgRequest<{
    ok: boolean
    status: number
    data?: ArrayBuffer
    error?: string
    headers?: Record<string, string>
  }>({
    path: `/api/v1/flashcards/assets/${encodeURIComponent(normalizedUuid)}/content` as AllowedPath,
    method: "GET",
    responseType: "arrayBuffer",
    returnResponse: true
  })
    .then((response) => {
      if (!response?.ok) {
        throw new Error(response?.error || `Asset fetch failed: ${response?.status ?? "unknown"}`)
      }
      if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function") {
        throw new Error("Object URLs are not supported in this environment.")
      }
      const headers = new Headers(response.headers || {})
      const blob = new Blob([response.data ?? new Uint8Array()], {
        type: headers.get("content-type") || "application/octet-stream"
      })
      const objectUrl = URL.createObjectURL(blob)
      const current = objectUrlCache.get(normalizedUuid)
      if (!current) {
        URL.revokeObjectURL(objectUrl)
        return objectUrl
      }
      current.objectUrl = objectUrl
      current.promise = undefined
      if (current.refCount <= 0) {
        URL.revokeObjectURL(objectUrl)
        objectUrlCache.delete(normalizedUuid)
      }
      return objectUrl
    })
    .catch((error) => {
      const current = objectUrlCache.get(normalizedUuid)
      if (current?.promise) {
        objectUrlCache.delete(normalizedUuid)
      }
      throw error
    })

  return await entry.promise
}

export function releaseFlashcardAssetObjectUrl(assetUuid: string): void {
  const normalizedUuid = String(assetUuid || "").trim()
  if (!normalizedUuid) return
  const entry = objectUrlCache.get(normalizedUuid)
  if (!entry) return
  entry.refCount -= 1
  if (entry.refCount > 0) return
  if (entry.objectUrl && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
    URL.revokeObjectURL(entry.objectUrl)
    objectUrlCache.delete(normalizedUuid)
    return
  }
  if (!entry.promise) {
    objectUrlCache.delete(normalizedUuid)
  }
}

export function clearFlashcardAssetObjectUrlCache(): void {
  objectUrlCache.forEach((entry, assetUuid) => {
    entry.refCount = 0
    if (entry.objectUrl && typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
      URL.revokeObjectURL(entry.objectUrl)
    }
    objectUrlCache.delete(assetUuid)
  })
}
