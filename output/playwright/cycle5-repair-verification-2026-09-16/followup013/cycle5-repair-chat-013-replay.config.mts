import { execFileSync } from 'node:child_process'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts'
const repo = '/Users/macbook-dev/Documents/GitHub/tldw_server2'
const suffixes = process.env.UAT013_REPLAY === 'restore' ? ['hooks/usePlaygroundSessionPersistence.tsx'] : ['components/Option/Playground/PlaygroundForm.tsx', 'hooks/usePlaygroundSessionPersistence.tsx']
const sources = new Map(suffixes.map(suffix => {
  const path = 'apps/packages/ui/src/' + suffix
  return [repo + '/' + path, execFileSync('git', ['show', '3c30685611:' + path], { cwd: repo, encoding: 'utf8' })]
}))
export default { ...base, plugins: [{ name: 'replay-frozen-013-source', enforce: 'pre', transform(_code, id) { const source = sources.get(id.split('?')[0]); return source ? { code: source, map: null } : undefined } }] }
