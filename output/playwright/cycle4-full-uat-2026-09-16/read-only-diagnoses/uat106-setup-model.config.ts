import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
const ui='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const frontend='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend'
const target=ui+'/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx'
export default {...base,resolve:{...base.resolve,alias:[
{find:'@plasmohq/storage/hook',replacement:frontend+'/extension/shims/plasmo-storage-hook.tsx'},
{find:'@plasmohq/storage',replacement:frontend+'/extension/shims/plasmo-storage.ts'},
{find:'@',replacement:ui+'/src'},{find:'~',replacement:ui+'/src'}
]},plugins:[{name:'uat106-wizard-model-boundary',enforce:'pre',transform(code,id){
if(id!==target)return
const insert=code.lastIndexOf('\n});')
if(insert<0)throw new Error('Expected wizard fixture describe ending')
return {code:code.slice(0,insert)+'\n'+fs.readFileSync('/private/tmp/uat106-setup-model-cases.txt','utf8')+code.slice(insert),map:null}
}}],test:{...base.test,setupFiles:[ui+'/vitest.setup.ts'],include:[target]}}
