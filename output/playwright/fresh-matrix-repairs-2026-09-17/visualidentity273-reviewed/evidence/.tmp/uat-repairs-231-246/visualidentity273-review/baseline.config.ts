import fs from 'node:fs'
import path from 'node:path'
import base from '../../../apps/packages/ui/vitest.config'
const panelPath=path.resolve(__dirname,'../../../apps/packages/ui/src/components/Common/VisualIdentity/VisualIdentityPackPanel.tsx')
const testPath=path.resolve(__dirname,'../../../apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityPackPanel.test.tsx')
const panel=fs.readFileSync(path.resolve(__dirname,'../visualidentity273/baseline-source/VisualIdentityPackPanel.tsx'),'utf8')
const test=fs.readFileSync(path.resolve(__dirname,'../visualidentity273/baseline-source/__tests__/VisualIdentityPackPanel.baseline.test.tsx'),'utf8')
export default {...base,plugins:[{name:'review-baseline-at-canonical-modules',enforce:'pre' as const,load(id:string){if(id.split('?')[0]===panelPath)return panel;if(id.split('?')[0]===testPath)return test}}]}
