import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts'
export default {
  ...base,
  plugins: [{
    name: 'uat152-read-only-baseline', enforce: 'pre' as const,
    load(id: string) {
      for (const file of ['tldw.tsx', 'TldwTimeoutSettings.tsx']) {
        if (id === '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Settings/' + file) {
          return fs.readFileSync('/private/tmp/cycle5-uat152-baseline-' + file, 'utf8')
        }
      }
    }
  }]
}
