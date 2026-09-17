import { defineConfig, mergeConfig } from '../../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from '../chat231-232-236-243/vitest.config'
import fs from 'node:fs'
import path from 'node:path'
export default mergeConfig(base,defineConfig({plugins:[{name:'review-actual-auth-lease',enforce:'pre',load(id){if(id.split('?')[0].endsWith('/hooks/chat/__tests__/useChatActions.character.integration.test.tsx'))return fs.readFileSync(path.resolve(__dirname,'actual-auth-lease-probe.tsx.txt'),'utf8')}}]}))
