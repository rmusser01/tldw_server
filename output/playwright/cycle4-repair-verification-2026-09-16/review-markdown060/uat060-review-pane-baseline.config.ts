import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
import { execFileSync } from 'node:child_process'
const root = '/Users/macbook-dev/Documents/GitHub/tldw_server2'
const source = 'apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx'
const test = 'apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx'
export default { ...base, plugins: [{ name: 'baseline-source', enforce: 'pre' as const, transform(_code: string, id: string) { return id === root + '/' + source ? { code: execFileSync('git', ['show', 'HEAD:' + source], { cwd: root, encoding: 'utf8' }), map: null } : null } }], test: { ...base.test, setupFiles: [root + '/apps/packages/ui/vitest.setup.ts'], include: [root + '/' + test] } }
