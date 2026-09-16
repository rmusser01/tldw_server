import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
const ui = '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const test = ui + '/src/hooks/__tests__/useMediaReadingProgress.test.tsx'
export default {
  ...base,
  plugins: [{ name: 'review-only-lifetime-controls', enforce: 'pre', transform(code, id) {
    if (id !== test) return
    const index = code.lastIndexOf('\n})')
    if (index < 0) throw new Error('Expected test describe boundary missing')
    return { code: code.slice(0, index) + '\n' + fs.readFileSync('/private/tmp/uat099-independent-lifetime-cases.txt', 'utf8') + code.slice(index), map: null }
  } }],
  test: { ...base.test, setupFiles: [ui + '/vitest.setup.ts'], include: [test] }
}
