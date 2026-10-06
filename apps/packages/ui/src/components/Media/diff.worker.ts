import { computeDiffSync, type DiffLine } from './line-diff'

type DiffWorkerRequest = {
  leftText: string
  rightText: string
}

type DiffWorkerResult = {
  type: 'result'
  lines: DiffLine[]
}

type DiffWorkerError = {
  type: 'error'
  message: string
}

self.onmessage = (event: MessageEvent<DiffWorkerRequest>) => {
  try {
    const request = event.data
    const result: DiffWorkerResult = {
      type: 'result',
      lines: computeDiffSync(request?.leftText || '', request?.rightText || '', {
        maxEditLength: 2048,
        timeout: 1000
      })
    }
    self.postMessage(result)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown diff worker error'
    const payload: DiffWorkerError = { type: 'error', message }
    self.postMessage(payload)
  }
}

export {}
