import type { ApiDataTableJobStatus } from "@/services/tldw/data-tables"

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer)
      signal?.removeEventListener("abort", onAbort)
      reject(new Error("Polling cancelled"))
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort)
      resolve()
    }, ms)
    if (signal?.aborted) onAbort()
    else signal?.addEventListener("abort", onAbort, { once: true })
  })

export const pollDataTableJob = async ({
  jobId,
  fetchJob,
  signal,
  intervalMs = 1000,
  timeoutMs = 5 * 60 * 1000
}: {
  jobId: number
  fetchJob: (jobId: number) => Promise<ApiDataTableJobStatus>
  signal?: AbortSignal
  intervalMs?: number
  timeoutMs?: number
}): Promise<ApiDataTableJobStatus> => {
  const start = Date.now()
  let pendingPublication: ApiDataTableJobStatus | undefined
  while (true) {
    if (signal?.aborted) {
      throw new Error("Polling cancelled")
    }
    let status: ApiDataTableJobStatus
    try {
      status = await fetchJob(jobId)
    } catch (error) {
      if (signal?.aborted) throw new Error("Polling cancelled")
      if (pendingPublication) return pendingPublication
      throw error
    }
    if (signal?.aborted) throw new Error("Polling cancelled")
    const state = String(status?.status || "").toLowerCase()
    const awaitingPublication = state === "completed" && status.result?.publication_status === "pending"
    if (awaitingPublication) pendingPublication = status
    if (!awaitingPublication && state && ["completed", "failed", "cancelled"].includes(state)) {
      return status
    }
    if (Date.now() - start > timeoutMs) {
      if (pendingPublication) return pendingPublication
      throw new Error("Job timed out")
    }
    await sleep(intervalMs, signal)
  }
}
