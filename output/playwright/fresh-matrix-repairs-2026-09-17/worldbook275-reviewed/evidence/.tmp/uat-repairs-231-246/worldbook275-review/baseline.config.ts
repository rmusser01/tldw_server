import fs from 'node:fs'
import path from 'node:path'
import base from '../../../apps/packages/ui/vitest.config'
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2'
const target=path.join(root,'apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx')
export default {...base, plugins:[{name:'review-only-baseline-manager',enforce:'pre',load(id:string){if(id.split('?')[0]===target)return fs.readFileSync(path.join(root,'.tmp/uat-repairs-231-246/worldbook275-review/baseline-manager.tsx'),'utf8')}}]}
