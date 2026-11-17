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

const check = (sources: Record<string, string>, baseline: string[][]) =>
  checkImportCycles(findImportCycles(fixture(sources)), baseline)

describe('shared UI import-cycle ratchet', () => {
  it('counts runtime cycles only, resolving aliases', () => {
    expect(findImportCycles(fixture(base))).toMatchObject({ cycles: [['a.ts', 'b.ts']], unresolved: [] })
  })
  it('passes at the baseline and when a baselined cycle shrinks', () => {
    expect(check(base, [['a.ts', 'b.ts']])).toEqual([])
    expect(check(base, [['a.ts', 'b.ts', 'gone.ts']])).toEqual([])
  })
  it('rejects a seeded new cycle, including one closed by a re-export', () => {
    const failures = check(
      { ...base, 'j.ts': 'export * from "./k"\nexport const j = 1', 'k.ts': 'import { j } from "./j"\nexport const k = j' },
      [['a.ts', 'b.ts']]
    )
    expect(failures).toEqual([expect.stringMatching(/^New or grown import cycle of 2 files.*\n {2}j\.ts {2}\(not in BASELINE\)\n {2}k\.ts/)])
  })
  it('rejects a baselined cycle that grew, naming the new member', () => {
    const failures = check(
      { ...base, 'b.ts': 'import { a } from "./a"\nimport "./l"\nexport const b = () => a', 'l.ts': 'import "./b"' },
      [['a.ts', 'b.ts']]
    )
    expect(failures).toEqual([expect.stringContaining('l.ts  (not in BASELINE)')])
  })
  it('rejects trading a baselined cycle for a new one of the same size', () => {
    const failures = check(
      {
        ...base,
        'b.ts': 'export const b = () => 1',
        'j.ts': 'import { k } from "./k"\nexport const j = k',
        'k.ts': 'import { j } from "./j"\nexport const k = j'
      },
      [['a.ts', 'b.ts']]
    )
    expect(failures).toEqual([
      expect.stringMatching(/^New or grown import cycle of 2 files/),
      expect.stringMatching(/^Stale baseline: this cycle is gone; delete its BASELINE entry in scripts\/check-ui-import-cycles\.mjs/)
    ])
  })
  it('rejects a stale baseline entry once its cycle is gone', () => {
    expect(check(base, [['a.ts', 'b.ts'], ['x.ts', 'y.ts']])).toEqual([
      expect.stringMatching(/^Stale baseline: .*\n {2}x\.ts\n {2}y\.ts$/)
    ])
  })
  it('rejects a local import that resolves to no file, but not assets or packages', () => {
    const sources = {
      ...base,
      'm.ts': 'import { z } from "./missing"\nimport "./style.css"\nimport logo from "@/logo.svg?url"\nimport "react"\nexport const m = [z, logo]'
    }
    expect(check(sources, [['a.ts', 'b.ts']])).toEqual([
      'm.ts imports "./missing", which resolves to no file: fix the path. An import this script cannot resolve is an edge it cannot check.'
    ])
  })
})
