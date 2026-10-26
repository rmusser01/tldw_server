// @vitest-environment node
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { checkImportCycles, findImportCycles } from '../check-ui-import-cycles.mjs'

const roots: string[] = []

function fixture(sources: Record<string, string>) {
  const uiRoot = mkdtempSync(path.join(tmpdir(), 'ui-import-cycles-'))
  roots.push(uiRoot)
  const tsconfig = {
    compilerOptions: { module: 'ESNext', moduleResolution: 'Bundler', jsx: 'react-jsx', baseUrl: './src', paths: { '@/*': ['./*'] } },
    include: ['src/**/*']
  }
  writeFileSync(path.join(uiRoot, 'tsconfig.json'), JSON.stringify(tsconfig))
  for (const [file, source] of Object.entries(sources)) {
    mkdirSync(path.dirname(path.join(uiRoot, 'src', file)), { recursive: true })
    writeFileSync(path.join(uiRoot, 'src', file), source)
  }
  return uiRoot
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

// One runtime cycle through an alias and a relative path, plus edges that compile away.
const base = {
  'a.ts': 'import { b } from "@/b"\nexport const a = () => b',
  'b.ts': 'import { a } from "./a"\nexport const b = () => a',
  'type-only/c.ts': 'import type { D } from "./d"\nexport const c = (d: D) => d',
  'type-only/d.ts': 'import { c } from "./c"\nexport type D = typeof c',
  'type-only/e.ts': 'import { type F } from "./f"\nexport const e = (f: F) => f',
  'type-only/f.ts': 'export type { E } from "./e"\nexport { type e as F } from "./e"',
  'lazy/g.ts': 'export const g = () => import("./h")',
  'lazy/h.ts': 'import { g } from "./g"\nexport const h = g',
  'ambient.d.ts': 'import { a } from "./a"\ndeclare global { const seeded: typeof a }'
}

describe('shared UI import-cycle ratchet', () => {
  it('counts runtime cycles only, resolving aliases', () => {
    expect(findImportCycles(fixture(base)).cycles).toEqual([['a.ts', 'b.ts']])
  })
  it('passes at the baseline', () => {
    const { cycles } = findImportCycles(fixture(base))
    expect(checkImportCycles(cycles, { cycles: 1, largestCycle: 2, filesInCycles: 2 }).failures).toEqual([])
  })
  it('rejects a seeded new cycle, including one closed by a re-export', () => {
    const { cycles } = findImportCycles(
      fixture({ ...base, 'j.ts': 'export * from "./k"\nexport const j = 1', 'k.ts': 'import { j } from "./j"\nexport const k = j' })
    )
    expect(checkImportCycles(cycles, { cycles: 1, largestCycle: 2, filesInCycles: 4 }).failures).toEqual([
      expect.stringMatching(/^2 import cycles, up from 1: /)
    ])
  })
  it('rejects a cycle that grew past the ceiling', () => {
    const { cycles } = findImportCycles(
      fixture({ ...base, 'b.ts': 'import { a } from "./a"\nimport "./l"\nexport const b = () => a', 'l.ts': 'import "./b"' })
    )
    expect(checkImportCycles(cycles, { cycles: 1, largestCycle: 2, filesInCycles: 3 }).failures).toEqual([
      expect.stringMatching(/^The largest import cycle grew from 2 to 3 files/)
    ])
  })
  it('rejects a smaller cycle that grew while staying under the largest', () => {
    const { cycles } = findImportCycles(
      fixture({ ...base, 'b.ts': 'import { a } from "./a"\nimport "./l"\nexport const b = () => a', 'l.ts': 'import "./b"' })
    )
    expect(checkImportCycles(cycles, { cycles: 1, largestCycle: 4, filesInCycles: 2 }).failures).toEqual([
      expect.stringMatching(/^3 files are in import cycles, up from 2/)
    ])
  })
  it('rejects a stale baseline once a cycle is gone', () => {
    const { cycles } = findImportCycles(fixture(base))
    expect(checkImportCycles(cycles, { cycles: 2, largestCycle: 2, filesInCycles: 4 }).failures).toEqual([
      expect.stringContaining('Lower BASELINE.cycles to 1 in scripts/check-ui-import-cycles.mjs')
    ])
  })
})
