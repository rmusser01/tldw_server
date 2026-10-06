import { afterEach, expect, it, vi } from 'vitest'
import {
  computeDiffSync,
  computeDiffWithWorker,
  shouldRequireSampling
} from '../diff-worker-client'

afterEach(() => vi.unstubAllGlobals())

it('keeps unchanged lines for an ordinary revision with more than 128 changed lines', () => {
  const left = Array.from({ length: 500 }, (_, i) => 'line ' + i)
  const right = left.map((line, i) => (i % 3 === 0 ? 'updated ' + i : line))
  const lines = computeDiffSync(left.join('\n'), right.join('\n'))
  expect(lines.filter((line) => line.type === 'same').length).toBe(333)
})

it('keeps common leading and trailing lines when the edit budget is exhausted', () => {
  const left = ['heading', ...Array.from({ length: 1200 }, (_, i) => 'old ' + i), 'ending']
  const right = ['heading', ...Array.from({ length: 1200 }, (_, i) => 'new ' + i), 'ending']
  const lines = computeDiffSync(left.join('\n'), right.join('\n'))
  expect(lines.filter((line) => line.type === 'same').map((line) => line.text)).toEqual([
    'heading',
    'ending'
  ])
  expect(lines.filter((line) => line.type !== 'add').map((line) => line.text)).toEqual(left)
  expect(lines.filter((line) => line.type !== 'del').map((line) => line.text)).toEqual(right)
})

it('allows the actual worker a larger bounded edit budget than the main thread', async () => {
  const left = Array.from({ length: 2500 }, (_, i) => 'line ' + i)
  const right = left.map((line, i) => (i < 1700 && i % 2 === 0 ? 'updated ' + i : line))
  const worker = { onmessage: null as ((event: MessageEvent) => void) | null, postMessage: vi.fn() }
  vi.stubGlobal('self', worker)
  await import('../diff.worker')
  worker.onmessage?.({
    data: { leftText: left.join('\n'), rightText: right.join('\n') }
  } as MessageEvent)
  const lines = worker.postMessage.mock.calls[0][0].lines
  expect(lines.filter((line: { type: string }) => line.type === 'same').length).toBe(1650)
})

it('reconstructs both sides of an ordinary line diff', () => {
  const lines = computeDiffSync('one\ntwo\nthree', 'one\nnew\nthree')
  expect(
    lines
      .filter((line) => line.type !== 'add')
      .map((line) => line.text)
      .join('\n')
  ).toBe('one\ntwo\nthree')
  expect(
    lines
      .filter((line) => line.type !== 'del')
      .map((line) => line.text)
      .join('\n')
  ).toBe('one\nnew\nthree')
  expect(lines.filter((line) => line.type === 'same').map((line) => line.text)).toEqual([
    'one',
    'three'
  ])
})

it('bounds short-line comparisons even below the character threshold', () => {
  const text = 'x\n'.repeat(70_000)
  expect(shouldRequireSampling(text, text)).toBe(true)
  // Stop the old implementation before it can exhaust the test process.
  const from = Array.from
  vi.spyOn(Array, 'from').mockImplementation(((
    input: ArrayLike<unknown>,
    mapper: (value: unknown, index: number) => unknown
  ) => {
    if (input.length > 10_000) throw new Error('Unsafe diff allocation')
    return from(input, mapper)
  }) as typeof Array.from)
  const lines = computeDiffSync(text, text)
  expect(lines.length).toBeLessThan(10_000)
  expect(lines.some((line) => line.text.includes('omitted'))).toBe(true)
})

it('terminates and rejects a pending worker on abort', async () => {
  const terminate = vi.fn()
  class PendingWorker {
    onmessage = null
    onerror = null
    terminate = terminate
    postMessage() {}
  }
  vi.stubGlobal('Worker', PendingWorker)
  const controller = new AbortController()
  const pending = computeDiffWithWorker('left', 'right', controller.signal)
  const rejection = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  controller.abort()
  expect(terminate).toHaveBeenCalledOnce()
  await rejection
})

it('terminates a worker when posting its request throws', async () => {
  const terminate = vi.fn()
  vi.stubGlobal(
    'Worker',
    class {
      terminate = terminate
      postMessage() {
        throw new Error('Cannot post')
      }
    }
  )
  await expect(computeDiffWithWorker('left', 'right')).rejects.toThrow('Cannot post')
  expect(terminate).toHaveBeenCalledOnce()
})
