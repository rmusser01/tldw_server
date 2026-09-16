import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config'
const target = '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/__tests__/components/notification-connectivity.integration.test.tsx'
export default {
  ...base,
  plugins: [...(base.plugins || []), {
    name: 'uat142-batched-read-only-independent-probes', enforce: 'pre',
    load(id: string) {
      if (id !== target) return
      const source = fs.readFileSync(target, 'utf8')
      const pos = source.lastIndexOf('\n})')
      return source.slice(0, pos) + '\n' + fs.readFileSync('/private/tmp/cycle5-uat142-batched-private-tests.txt', 'utf8') + source.slice(pos)
    }
  }]
}
