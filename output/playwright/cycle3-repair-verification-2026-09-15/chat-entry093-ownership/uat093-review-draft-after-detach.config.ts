import { readFileSync } from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config'
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2'
const target=root+'/apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx'
export default {...base, plugins:[...(base.plugins??[]),{name:'review-draft-after-detach',enforce:'pre' as const,load(id:string){if(id===target)return readFileSync('/private/tmp/uat093-review-draft-after-detach-fixture.tsx','utf8')}}],test:{...base.test,setupFiles:[root+'/apps/tldw-frontend/vitest.setup.ts'],include:[target]}}
