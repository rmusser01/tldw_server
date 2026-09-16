import { execFileSync } from 'node:child_process'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts'
const root = '/Users/macbook-dev/Documents/GitHub/tldw_server2'
const target = 'apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts'
const baseline = execFileSync('git', ['show', '045a953883b5f1809c61d7c171eda83e05450a9b:' + target], { cwd: root, encoding: 'utf8' })
export default { ...base, root: root + '/apps/packages/ui', plugins: [{ name: 'uat118-before-mime-repair', enforce: 'pre', transform(_code, id) { return id.split('?')[0] === root + '/' + target ? { code: baseline, map: null } : null } }], test: { ...base.test, include: ['src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx'] } }
