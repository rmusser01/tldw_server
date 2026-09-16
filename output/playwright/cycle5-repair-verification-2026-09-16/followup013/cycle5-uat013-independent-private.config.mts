import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
const target = '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx'
export default {
  ...base,
  plugins: [...(base.plugins || []), {
    name: 'uat013-independent-read-only-probe', enforce: 'pre',
    load(id: string) {
      if (id !== target) return
      const source = fs.readFileSync(target, 'utf8')
      const marker = '  it("allows a later requested restore on the same hook'
      const pos = source.indexOf(marker)
      if (pos < 0) throw new Error('Missing insertion boundary')
      return source.slice(0,pos) + fs.readFileSync('/private/tmp/cycle5-uat013-independent-private-test.txt','utf8') + '\n' + source.slice(pos)
    }
  }]
}
