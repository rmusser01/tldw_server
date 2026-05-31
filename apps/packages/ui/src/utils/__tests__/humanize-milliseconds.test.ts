import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { humanizeMilliseconds } from '../humanize-milliseconds'

describe('humanizeMilliseconds', () => {
    it.each([
        [999, '999ms'],
        [1000, '1s'],
        [59_999, '59s'],
        [60_000, '1m'],
        [3_599_999, '59m'],
        [3_600_000, '1h'],
        [86_399_999, '23h'],
        [86_400_000, '1d'],
        [172_800_000, '2d'],
    ])('formats %i milliseconds as %s', (milliseconds, expected) => {
        expect(humanizeMilliseconds(milliseconds)).toBe(expected)
    })

    it('does not depend on dayjs for display-only duration formatting', () => {
        const testDir = dirname(fileURLToPath(import.meta.url))
        const source = readFileSync(resolve(testDir, '../humanize-milliseconds.ts'), 'utf8')

        expect(source).not.toContain('dayjs')
    })
})
