import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const srcRoot = resolve(__dirname, '../../..')
const dayjsImportPattern =
  /^\s*import\s+(?:type\s+)?(?:.+?\s+from\s+)?['"]dayjs(?:\/[^'"]*)?['"]/

const collectSourceFiles = (directory: string): string[] => {
  return readdirSync(directory).flatMap((entry) => {
    const fullPath = join(directory, entry)
    const stats = statSync(fullPath)

    if (stats.isDirectory()) {
      return collectSourceFiles(fullPath)
    }

    return /\.(ts|tsx)$/.test(entry) ? [fullPath] : []
  })
}

const collectDayjsImports = (): string[] => {
  return collectSourceFiles(srcRoot).flatMap((filePath) => {
    const relativePath = relative(srcRoot, filePath).replace(/\\/g, '/')
    return readFileSync(filePath, 'utf8')
      .split(/\r?\n/)
      .flatMap((line) => (dayjsImportPattern.test(line) ? [`${relativePath}:${line.trim()}`] : []))
  })
}

describe('FilterPanel dayjs dependency cleanup', () => {
  it('leaves only the deferred ReadingList and Items dayjs import surfaces', () => {
    const dayjsImports = collectDayjsImports()
    const importCountsByFile = dayjsImports.reduce<Record<string, number>>(
      (counts, importMatch) => {
        const [filePath] = importMatch.split(':')
        counts[filePath] = (counts[filePath] ?? 0) + 1
        return counts
      },
      {}
    )

    expect(dayjsImports).toHaveLength(4)
    expect(importCountsByFile).toEqual({
      'components/Option/Collections/ReadingList/ReadingItemsList.tsx': 2,
      'components/Option/Items/ItemsWorkspace.tsx': 2
    })
    expect(dayjsImports).not.toEqual(
      expect.arrayContaining([
        expect.stringMatching(/^components\/Media\/FilterPanel\.tsx:/)
      ])
    )
  })
})
