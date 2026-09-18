import fs from 'node:fs'
import path from 'node:path'
import base from '../../../apps/packages/ui/vitest.config'
const testPath=path.resolve(__dirname,'../../../apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityPackPanel.test.tsx')
const reviewTest=fs.readFileSync(path.resolve(__dirname,'controls.review.test.tsx'),'utf8')
export default {...base,plugins:[{name:'review-controls-at-canonical-module',enforce:'pre' as const,load(id:string){if(id.split('?')[0]===testPath)return reviewTest}}]}
