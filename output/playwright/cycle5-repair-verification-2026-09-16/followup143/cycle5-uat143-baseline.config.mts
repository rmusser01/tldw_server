import { execFileSync } from 'node:child_process'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts'
const root = '/Users/macbook-dev/Documents/GitHub/tldw_server2'
const changed = execFileSync('git', ['diff', '--name-only', '3c30685611', '--', 'apps/packages/ui/src'], { cwd: root, encoding: 'utf8' }).trim().split('\n').filter(p => !p.includes('/__tests__/') && /\.[tj]sx?$/.test(p))
const sources = new Map(changed.map(p => [root + '/' + p, execFileSync('git', ['show', '3c30685611:' + p], { cwd: root, encoding: 'utf8' })]))
export default { ...base, plugins: [{ name: 'uat143-baseline-read-only', enforce: 'pre', transform(code, id) {
  const file = id.split('?')[0]
  if (/PlaygroundForm\.(composer-options|llamacpp-controls)\.guard\.test\.ts$/.test(file)) {
    return { code: code.replace(/import fs from "node:fs";?/, `import { execFileSync } from 'node:child_process'; const fs = { readFileSync: (p, _encoding) => execFileSync('git', ['show', '3c30685611:' + p.slice(${JSON.stringify(root + '/')}.length)], { cwd: ${JSON.stringify(root)}, encoding: 'utf8' }) }`), map: null }
  }
  const original = sources.get(file)
  return original ? { code: original, map: null } : undefined
} }] }
