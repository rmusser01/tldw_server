import { MAX_EXPORT_BYTES, MAX_REPO_ENTRIES } from "./exportPolicy"
import type {
  IProvider,
  RepoFileContent,
  RepoTreeNode
} from "./providers/types"

type FetchFilesWithConcurrencyOptions = {
  nodes: RepoTreeNode[]
  provider: Pick<IProvider, "fetchFile">
  limit?: number
  signal?: AbortSignal
  onProgress?: (completed: number, total: number) => void
}

export async function fetchFilesWithConcurrency({
  nodes,
  provider,
  limit = 5,
  onProgress,
  signal
}: FetchFilesWithConcurrencyOptions): Promise<RepoFileContent[]> {
  signal?.throwIfAborted()
  if (nodes.length > MAX_REPO_ENTRIES)
    throw new Error("Repository exceeds 5000 files")
  if (nodes.length === 0) {
    onProgress?.(0, 0)
    return []
  }

  const safeLimit = Number.isFinite(limit) ? Math.max(1, Math.floor(limit)) : 5
  const controller = new AbortController()
  const abort = () => controller.abort()
  signal?.addEventListener("abort", abort, { once: true })
  let totalBytes = 0
  const result: RepoFileContent[] = new Array(nodes.length)
  let cursor = 0
  let completed = 0

  const worker = async () => {
    while (true) {
      controller.signal.throwIfAborted()
      const index = cursor
      cursor += 1
      if (index >= nodes.length) {
        return
      }
      const content = await provider.fetchFile(nodes[index], controller.signal)
      controller.signal.throwIfAborted()
      totalBytes += new TextEncoder().encode(content.text).byteLength
      if (totalBytes > MAX_EXPORT_BYTES)
        throw new Error("Selected text exceeds 10 MiB")
      result[index] = content
      completed += 1
      onProgress?.(completed, nodes.length)
    }
  }

  const workers = Array.from(
    { length: Math.min(safeLimit, nodes.length) },
    () => worker()
  )
  try {
    await Promise.all(workers)
    return result
  } finally {
    controller.abort()
    signal?.removeEventListener("abort", abort)
  }
}
