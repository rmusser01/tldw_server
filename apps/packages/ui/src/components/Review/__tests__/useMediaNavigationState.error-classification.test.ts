import { describe, expect, it } from 'vitest'

import { shouldReportMediaDetailFetchError } from '../mediaDetailError'

describe('media detail error reporting', () => {
  it('does not report expected missing or trashed responses as runtime errors', () => {
    expect(
      shouldReportMediaDetailFetchError(
        Object.assign(new Error('missing'), { status: 404 })
      )
    ).toBe(false)
    expect(
      shouldReportMediaDetailFetchError(
        Object.assign(new Error('trashed'), { status: 410 })
      )
    ).toBe(false)
  })

  it('continues reporting unexpected detail failures', () => {
    expect(
      shouldReportMediaDetailFetchError(
        Object.assign(new Error('server failure'), { status: 500 })
      )
    ).toBe(true)
  })
})
