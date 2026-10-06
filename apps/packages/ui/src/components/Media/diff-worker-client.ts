import { type DiffLine } from './line-diff'
export {
  computeDiffSync,
  shouldRequireSampling,
  shouldUseWorkerDiff,
  sampleTextForDiff,
  DIFF_HARD_CHAR_THRESHOLD,
  DIFF_SYNC_LINE_THRESHOLD,
  DIFF_SAMPLED_CHAR_BUDGET,
  type DiffLine
} from './line-diff'

type DiffWorkerRequest = {
  leftText: string
  rightText: string
}

type DiffWorkerResultMessage = {
  type: 'result'
  lines: DiffLine[]
}

type DiffWorkerErrorMessage = {
  type: 'error'
  message?: string
}

type DiffWorkerResponse = DiffWorkerResultMessage | DiffWorkerErrorMessage

export const createDiffWorker = (): Worker =>
  new Worker(new URL('./diff.worker.ts', import.meta.url), { type: 'module' })

export const computeDiffWithWorker = async (
  leftText: string,
  rightText: string,
  signal?: AbortSignal
): Promise<DiffLine[]> => {
  return await new Promise<DiffLine[]>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Diff cancelled', 'AbortError'))
      return
    }
    let settled = false
    const worker = createDiffWorker()

    const finalize = (fn: () => void) => {
      if (settled) return
      settled = true
      signal?.removeEventListener('abort', abort)
      worker.onmessage = null
      worker.onerror = null
      try {
        worker.terminate()
      } catch {
        // Ignore worker termination issues.
      }
      fn()
    }

    const abort = () => finalize(() => reject(new DOMException('Diff cancelled', 'AbortError')))
    signal?.addEventListener('abort', abort, { once: true })

    worker.onmessage = (event: MessageEvent<DiffWorkerResponse>) => {
      const payload = event.data
      if (!payload || typeof payload !== 'object') {
        finalize(() => reject(new Error('Diff worker returned invalid payload')))
        return
      }
      if (payload.type === 'result') {
        finalize(() => resolve(Array.isArray(payload.lines) ? payload.lines : []))
        return
      }
      if (payload.type === 'error') {
        finalize(() => reject(new Error(payload.message || 'Diff worker failed')))
        return
      }
      finalize(() => reject(new Error('Diff worker returned unknown response type')))
    }

    worker.onerror = () => {
      finalize(() => reject(new Error('Diff worker crashed')))
    }

    const request: DiffWorkerRequest = {
      leftText: String(leftText || ''),
      rightText: String(rightText || '')
    }
    try {
      worker.postMessage(request)
    } catch (error) {
      finalize(() => reject(error))
    }
  })
}
