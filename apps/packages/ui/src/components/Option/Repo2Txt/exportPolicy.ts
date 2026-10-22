export const MAX_FILE_BYTES = 1024 * 1024
export const MAX_EXPORT_BYTES = 10 * MAX_FILE_BYTES
export const MAX_REPO_ENTRIES = 5000

export function validateRepoPath(path: string): void {
  if (
    !path ||
    path.startsWith("/") ||
    Array.from(path).some(
      (char) => char === "\\" || char === ":" || char.charCodeAt(0) < 32
    ) ||
    path.split("/").some((part) => part === ".." || part === "." || !part)
  ) {
    throw new Error(`Unsafe repository path: ${path}`)
  }
}

export function validateText(text: string, path: string): string {
  if (new TextEncoder().encode(text).byteLength > MAX_FILE_BYTES) {
    throw new Error(`File exceeds 1 MiB: ${path}`)
  }
  if (
    Array.from(text).some((char) => {
      const code = char.charCodeAt(0)
      return code <= 8 || (code >= 14 && code <= 31)
    })
  ) {
    throw new Error(`Binary file cannot be exported as text: ${path}`)
  }
  return text
}

export function decodeText(bytes: Uint8Array, path: string): string {
  try {
    return validateText(
      new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes),
      path
    )
  } catch (error) {
    if (error instanceof TypeError)
      throw new Error(`File is not valid UTF-8 text: ${path}`)
    throw error
  }
}

export async function readBoundedResponse(
  response: Response,
  limit: number,
  signal?: AbortSignal
): Promise<Uint8Array> {
  signal?.throwIfAborted()
  const length = Number(response.headers.get("content-length"))
  if (length > limit)
    throw new Error("Repository response exceeds the size limit")
  const reader = response.body?.getReader()
  if (!reader) throw new Error("Repository response has no readable body")
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      signal?.throwIfAborted()
      const { done, value } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > limit)
        throw new Error("Repository response exceeds the size limit")
      chunks.push(value)
    }
    signal?.throwIfAborted()
    const bytes = new Uint8Array(size)
    let offset = 0
    for (const chunk of chunks) {
      bytes.set(chunk, offset)
      offset += chunk.byteLength
    }
    return bytes
  } finally {
    await reader.cancel()
    reader.releaseLock()
  }
}
