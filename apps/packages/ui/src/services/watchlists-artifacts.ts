import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type WatchlistArtifactErrorKind = "missing" | "auth" | "network" | "unsafe"

export class WatchlistArtifactError extends Error {
  constructor(
    public readonly kind: WatchlistArtifactErrorKind,
    message: string,
    public readonly status?: number
  ) {
    super(message)
    this.name = "WatchlistArtifactError"
  }
}

const apiArtifactPath = (value: string): AllowedPath => {
  try {
    if (!value.startsWith("/")) throw new Error("absolute-or-relative-url")
    const parsed = new URL(value, "https://tldw.invalid")
    const decodedPath = decodeURIComponent(parsed.pathname)
    if (
      parsed.origin !== "https://tldw.invalid" ||
      !decodedPath.startsWith("/api/v1/") ||
      decodedPath.split("/").includes("..") ||
      decodedPath.includes("\\")
    ) {
      throw new Error("unsafe-artifact-path")
    }
    return `${parsed.pathname}${parsed.search}` as AllowedPath
  } catch {
    throw new WatchlistArtifactError("unsafe", "Unsafe artifact location")
  }
}

const artifactError = (error: unknown): WatchlistArtifactError => {
  if (error instanceof WatchlistArtifactError) return error
  const status = Number((error as { status?: unknown } | null)?.status)
  if (status === 404) return new WatchlistArtifactError("missing", "Artifact not found", status)
  if (status === 401 || status === 403) {
    return new WatchlistArtifactError("auth", "Artifact access was denied", status)
  }
  return new WatchlistArtifactError(
    "network",
    error instanceof Error ? error.message : "Artifact request failed",
    Number.isFinite(status) ? status : undefined
  )
}

export const fetchWatchlistArtifactBlob = async (
  path: string,
  options: { signal?: AbortSignal; mimeType?: string } = {}
): Promise<Blob> => {
  try {
    const data = await bgRequest<ArrayBuffer | ArrayBufferView | Blob>({
      path: apiArtifactPath(path),
      method: "GET",
      noAuth: false,
      abortSignal: options.signal,
      responseType: "arrayBuffer"
    })
    if (data instanceof Blob) return data
    if (ArrayBuffer.isView(data)) {
      const copy = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer
      return new Blob([copy], { type: options.mimeType })
    }
    return new Blob([data], { type: options.mimeType })
  } catch (error) {
    throw artifactError(error)
  }
}

export const fetchWatchlistArtifactText = async (
  path: string,
  options: { signal?: AbortSignal } = {}
): Promise<string> => {
  try {
    return await bgRequest<string>({
      path: apiArtifactPath(path),
      method: "GET",
      noAuth: false,
      abortSignal: options.signal,
      responseType: "text"
    })
  } catch (error) {
    throw artifactError(error)
  }
}

export const createWatchlistArtifactObjectUrl = (blob: Blob): string =>
  URL.createObjectURL(blob)

export const revokeWatchlistArtifactObjectUrl = (url: string | null | undefined): void => {
  if (url) URL.revokeObjectURL(url)
}
