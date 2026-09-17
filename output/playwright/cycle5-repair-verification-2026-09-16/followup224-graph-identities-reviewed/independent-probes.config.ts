import { defineConfig, mergeConfig } from '../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from '../../apps/tldw-frontend/vitest.config'
import path from 'node:path'
import fs from 'node:fs'
const root=path.resolve(__dirname,'../..')
const target=path.join(root,'apps/packages/ui/src/services/tldw/__tests__/note-graph-identities.test.ts')
export default mergeConfig(base,defineConfig({root:path.join(root,'apps/tldw-frontend'),plugins:[{name:'independent-uat224-cases',enforce:'pre',load(id){return id.split('?')[0]===target ? fs.readFileSync(target,'utf8')+'\n'+fs.readFileSync(path.join(__dirname,'counterexamples.ts.txt'),'utf8') : null}}]}))
