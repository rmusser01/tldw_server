import { defineConfig, mergeConfig } from '../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from '../../apps/tldw-frontend/vitest.config'
import path from 'node:path'
import fs from 'node:fs'
const root = path.resolve(__dirname, '../..')
const files = new Map([[
 path.join(root, 'apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx'),
 path.join(__dirname, 'pre-lifetime-review/review-snapshot/apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx')
]])
export default mergeConfig(base, defineConfig({
  root: path.join(root, 'apps/tldw-frontend'),
  plugins: [{ name: 'read-only-uat221-pre-lifetime', enforce: 'pre', load(id) {
    const snapshot = files.get(id.split('?')[0])
    return snapshot ? fs.readFileSync(snapshot, 'utf8') : null
  }}]
}))
