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
  while (true) {
    if (signal?.aborted) {
      throw new Error("Polling cancelled")
    }
    const status = await fetchJob(jobId)
    if (signal?.aborted) throw new Error("Polling cancelled")
    const state = String(status?.status || "").toLowerCase()
    if (state && ["completed", "failed", "cancelled"].includes(state)) {
      return status
    }
    if (Date.now() - start > timeoutMs) {
      throw new Error("Job timed out")
    }
    await sleep(intervalMs, signal)
  }
}
