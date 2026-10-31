// @vitest-environment node
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { checkSharedHooks } from '../check-shared-hooks.mjs'

const roots: string[] = []
const realConfig = path.resolve('eslint.config.mjs')
const valid = 'export default function Example() { return <div>Ready</div> }'

function fixture(source = valid, override = '') {
  const appsRoot = mkdtempSync(path.join(tmpdir(), 'shared-hooks-'))
  roots.push(appsRoot)
  for (const scope of ['packages/ui/src', 'tldw-frontend/pages']) {
    mkdirSync(path.join(appsRoot, scope), { recursive: true })
    writeFileSync(path.join(appsRoot, scope, 'Example.tsx'), source)
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

describe('shared hook enforcement', () => {
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
  it('leaves unrelated rules to the existing full lint gate', async () => {
    const result = await checkSharedHooks(
      fixture('export default function Example() { 1; return <div /> }')
    )
    expect(result.failures).toEqual([])
    expect(result.otherErrorCount).toBeGreaterThan(0)
  })
})
