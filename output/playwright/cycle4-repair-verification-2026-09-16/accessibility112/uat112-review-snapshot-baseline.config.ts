import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
import { execFileSync } from 'node:child_process'
const root = '/Users/macbook-dev/Documents/GitHub/tldw_server2'
const paths = ['apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx', 'apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx']
export default { ...base, plugins: [{ name: 'baseline-source', enforce: 'pre' as const, transform(_code: string, id: string) { const path = paths.find(p => id === root + '/' + p); return path ? { code: execFileSync('git', ['show', 'HEAD:' + path], { cwd: root, encoding: 'utf8' }), map: null } : null } }], test: { ...base.test, setupFiles: [root + '/apps/packages/ui/vitest.setup.ts'], include: [root + '/' + paths[1]], testNamePattern: 'matches baseline snapshot for active review state' } }
