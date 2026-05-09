import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { humanizeMilliseconds } from '../humanize-milliseconds'

describe('humanizeMilliseconds', () => {
    it('preserves compact duration suffixes across existing thresholds', () => {
        expect(humanizeMilliseconds(999)).toBe('999ms')
        expect(humanizeMilliseconds(1000)).toBe('1s')
        expect(humanizeMilliseconds(59_999)).toBe('59s')
        expect(humanizeMilliseconds(60_000)).toBe('1m')
        expect(humanizeMilliseconds(3_599_999)).toBe('59m')
        expect(humanizeMilliseconds(3_600_000)).toBe('1h')
        expect(humanizeMilliseconds(86_399_999)).toBe('23h')
        expect(humanizeMilliseconds(86_400_000)).toBe('1d')
        expect(humanizeMilliseconds(172_800_000)).toBe('2d')
    })

    it('does not depend on dayjs for display-only duration formatting', () => {
        const testDir = dirname(fileURLToPath(import.meta.url))
        const source = readFileSync(resolve(testDir, '../humanize-milliseconds.ts'), 'utf8')

        expect(source).not.toContain('dayjs')
    })
})
