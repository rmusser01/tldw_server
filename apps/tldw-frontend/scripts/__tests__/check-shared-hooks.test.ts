// @vitest-environment node
import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { checkSharedHooks } from '../check-shared-hooks.mjs'
import { frozenUiFilePaths } from '../../eslint.config.mjs'

const roots: string[] = []
const realConfig = path.resolve('eslint.config.mjs')
const valid = 'export default function Example() { return <div>Ready</div> }'

const scopeFiles = ['packages/ui/src/Example.tsx', 'tldw-frontend/pages/Example.tsx']
// One `no-empty` error per block; the real config makes it an error.
const emptyBlocks = (count: number) =>
  `export default function Example({ value }) { ${'if (value) {} '.repeat(count)}return <div /> }`

function fixture(source = valid, override = '', suppressions?: Record<string, Record<string, { count: number }>>) {
  const appsRoot = mkdtempSync(path.join(tmpdir(), 'shared-hooks-'))
  roots.push(appsRoot)
  for (const file of scopeFiles) {
    mkdirSync(path.dirname(path.join(appsRoot, file)), { recursive: true })
    writeFileSync(path.join(appsRoot, file), source)
  }
  if (suppressions) {
    writeFileSync(
      path.join(appsRoot, 'tldw-frontend/eslint-suppressions.packages-ui.json'),
      JSON.stringify(suppressions)
    )
  }
  const configFile = path.join(appsRoot, 'eslint.config.mjs')
  writeFileSync(
    configFile,
    `import config from ${JSON.stringify(pathToFileURL(realConfig).href)}; export default [...config${override ? `, ${override}` : ''}];`
  )
  return { appsRoot, configFile }
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

describe('shared UI lint and hook enforcement', () => {
  it('leaves nonfatal unrelated unused-disable warnings to full lint', async () => {
    const result = await checkSharedHooks(
      fixture(`// eslint-disable-next-line no-console\n${valid}`)
    )
    expect(result.failures).toEqual([])
  })
  it('checks shared sources and WebUI pages with the real configuration', async () => {
    expect(await checkSharedHooks(fixture())).toMatchObject({ fileCount: 2, failures: [] })
  })
  it.each([
    ['purity', 'export default function Example() { return <div>{Date.now()}</div> }'],
    [
      'static-components',
      'export default function Example({ value }) { const Child = () => <div>{value}</div>; return <Child /> }'
    ],
    [
      'use-memo',
      "import { useMemo } from 'react'; export default function Example({ value }) { const result = useMemo(() => value, [JSON.stringify(value)]); return <div>{result}</div> }"
    ]
  ])('rejects a real %s violation in both scopes', async (rule, source) => {
    const result = await checkSharedHooks(fixture(source))
    expect(
      result.failures.filter((failure) => failure.ruleId === `react-hooks/${rule}`)
    ).toHaveLength(2)
  })
  it('rejects parser errors even though they have no rule identifier', async () => {
    const result = await checkSharedHooks(fixture('export default function Example( {'))
    expect(result.failures).toHaveLength(2)
    expect(result.failures.every((failure) => 'fatal' in failure && failure.fatal)).toBe(true)
  })
  it('rejects invalid ESLint configuration', async () => {
    await expect(
      checkSharedHooks(fixture(valid, '{ rules: { "unknown/rule": "error" } }'))
    ).rejects.toThrow()
  })
  it('rejects unknown rule definitions in inline configuration', async () => {
    const result = await checkSharedHooks(fixture(`/* eslint unknown-rule: error */\n${valid}`))
    expect(result.failures).toHaveLength(2)
  })
  it('rejects rules disabled by a file-specific override', async () => {
    const result = await checkSharedHooks(
      fixture(valid, '{ files: ["packages/ui/**/*.tsx"], rules: { "react-hooks/purity": "off" } }')
    )
    expect(result.failures).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          ruleId: 'react-hooks/purity',
          message: expect.stringContaining('must remain enabled as an error')
        })
      ])
    )
  })
  it('rejects silently ignored source files', async () => {
    const result = await checkSharedHooks(fixture(valid, '{ ignores: ["packages/ui/**"] }'))
    expect(result.failures).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ message: expect.stringContaining('not covered by ESLint') })
      ])
    )
  })
  it('rejects an empty required scope', async () => {
    const options = fixture()
    rmSync(path.join(options.appsRoot, 'packages/ui/src/Example.tsx'))
    await expect(checkSharedHooks(options)).rejects.toThrow('No source files')
  })
  it('rejects any other rule error that the suppressions file does not freeze', async () => {
    const result = await checkSharedHooks(fixture(emptyBlocks(1)))
    expect(result.failures).toHaveLength(2)
    expect(result.failures[0]).toMatchObject({
      ruleId: 'no-empty',
      hint: expect.stringContaining('eslint-disable-next-line no-empty -- <reason>')
    })
  })
  it('passes errors frozen by ESLint bulk suppressions', async () => {
    const frozen = { 'no-empty': { count: 2 } }
    const result = await checkSharedHooks(
      fixture(emptyBlocks(2), '', Object.fromEntries(scopeFiles.map((file) => [file, frozen])))
    )
    expect(result).toMatchObject({ failures: [], suppressedCount: 4 })
  })
  it('rejects every error of a rule whose frozen count grew', async () => {
    const result = await checkSharedHooks(
      fixture(emptyBlocks(2), '', { [scopeFiles[0]]: { 'no-empty': { count: 1 } } })
    )
    expect(result.failures.filter((failure) => failure.ruleId === 'no-empty')).toHaveLength(4)
  })
  it('rejects stale suppressions for fixed errors and for files no longer linted', async () => {
    const result = await checkSharedHooks(
      fixture(valid, '', {
        [scopeFiles[0]]: { 'no-empty': { count: 1 } },
        'packages/ui/src/Removed.tsx': { 'no-empty': { count: 1 } }
      })
    )
    expect(result.failures.map((failure) => failure.message)).toEqual([
      expect.stringMatching(/^Stale suppression: 0 errors left of 1 frozen; run `bun run lint:ui --prune-suppressions`/),
      expect.stringMatching(/^Stale suppression: file is gone or no longer linted/)
    ])
  })
  it('never lets the suppressions file freeze a hook rule', async () => {
    const result = await checkSharedHooks(
      fixture('export default function Example() { return <div>{Date.now()}</div> }', '', {
        [scopeFiles[0]]: { 'react-hooks/purity': { count: 1 } }
      })
    )
    expect(result.failures.map((failure) => failure.message)).toEqual(
      expect.arrayContaining([expect.stringContaining('must not freeze hook rules')])
    )
    expect(result.failures.filter((failure) => failure.ruleId === 'react-hooks/purity')).toHaveLength(3)
  })
  it('skips the vendored public assets that the real configuration ignores', async () => {
    const options = fixture()
    mkdirSync(path.join(options.appsRoot, 'packages/ui/src/public'))
    writeFileSync(path.join(options.appsRoot, 'packages/ui/src/public/vendor.min.js'), 'var x = ; if (')
    expect(await checkSharedHooks(options)).toMatchObject({ fileCount: 2, failures: [] })
  })
})

// Seeded violations of the TASK-13950.10 boundary rules, linted with the real configuration.
async function lintSeeds(seeds: Record<string, string>) {
  const options = fixture()
  for (const [file, source] of Object.entries(seeds)) {
    mkdirSync(path.dirname(path.join(options.appsRoot, file)), { recursive: true })
    writeFileSync(path.join(options.appsRoot, file), source)
  }
  const { failures } = await checkSharedHooks(options)
  return Object.keys(seeds).map((file) =>
    failures
      .filter((failure) => failure.filePath === path.join(options.appsRoot, file))
      .map((failure) => failure.ruleId)
  )
}

describe('shared UI boundary rules', () => {
  it.each([
    ['hooks importing components', 'packages/ui/src/hooks/useSeed.ts', 'import { Foo } from "@/components/Foo"\nexport const useSeed = () => Foo', 'no-restricted-imports'],
    ['utils re-exporting components', 'packages/ui/src/utils/seed.ts', 'export { Foo } from "../components/Foo"', 'no-restricted-imports'],
    ['store importing components lazily', 'packages/ui/src/store/seed.ts', 'export const load = () => import("~/components/Foo")', 'no-restricted-syntax'],
    ['components importing entries', 'packages/ui/src/components/Seed.tsx', 'import { Prompt } from "~/entries/shared/route-leave-prompt"\nexport default Prompt', 'no-restricted-imports'],
    ['routes importing background-proxy', 'packages/ui/src/routes/seed.tsx', 'import { bgRequest } from "@/services/background-proxy"\nexport default bgRequest', 'no-restricted-imports'],
    ['hooks importing background-proxy lazily', 'packages/ui/src/hooks/useLazy.ts', 'export const load = () => import("../services/background-proxy")', 'no-restricted-syntax'],
    ['a component hard-coding /api/v1/', 'packages/ui/src/components/Seed.tsx', 'export const url = (base: string) => `${base}/api/v1/media`', 'no-restricted-syntax'],
    ['an API path cast as any', 'packages/ui/src/services/seed.ts', 'export const path = (id: string) => `/api/v1/notes/${id}` as any', 'no-restricted-syntax']
  ])('rejects %s', async (_name, file, source, ruleId) => {
    expect(await lintSeeds({ [file]: source })).toEqual([[ruleId]])
  })
  it('leaves tests, services, prose and type-level paths alone', async () => {
    const results = await lintSeeds({
      'packages/ui/src/components/__tests__/Seed.test.tsx':
        'import { bgRequest } from "@/services/background-proxy"\nexport const call = () => bgRequest({ path: "/api/v1/media" as any })',
      'packages/ui/src/services/seed.ts': 'export { bgRequest } from "./background-proxy"\nexport const path = "/api/v1/media"',
      'packages/ui/src/components/Seed.tsx':
        'export type NotePath = `/api/v1/notes/${string}`\nexport const hint = "This server does not advertise /api/v1/notes."'
    })
    expect(results).toEqual([[], [], []])
  })
  it('rejects growth of a frozen chat file and of its largest function', async () => {
    const body = Array.from({ length: 3300 }, (_, index) => `  const v${index} = ${index}`).join('\n')
    const [ruleIds] = await lintSeeds({
      'packages/ui/src/routes/sidepanel-chat.tsx': `export function Big() {\n${body}\n  return v0\n}`
    })
    expect(ruleIds.sort()).toEqual(['max-lines', 'max-lines-per-function'])
  })
  it('freezes only files that still exist', () => {
    expect(frozenUiFilePaths).toHaveLength(5)
    expect(frozenUiFilePaths.filter((file) => !existsSync(path.resolve('..', file)))).toEqual([])
  })
})
