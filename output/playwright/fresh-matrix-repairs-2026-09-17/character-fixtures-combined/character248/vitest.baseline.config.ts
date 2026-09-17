import { defineConfig, mergeConfig } from '../../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from './vitest.config'
import fs from 'node:fs'
import path from 'node:path'
const ui = path.resolve(__dirname, '../../../apps/packages/ui/src')
const files = ['services/tldw/domains/chat-rag.ts', 'services/tldw/service-prompt-scope-error.ts', 'hooks/chat/useChatActions.ts']
export default mergeConfig(base, defineConfig({ plugins: [{
  name: 'uat-chat-original-source', enforce: 'pre',
  load(id) {
    const relative = path.relative(ui, id.split('?')[0])
    if (files.includes(relative)) return fs.readFileSync(path.resolve(__dirname, 'baseline', relative), 'utf8')
  }
}] }))
