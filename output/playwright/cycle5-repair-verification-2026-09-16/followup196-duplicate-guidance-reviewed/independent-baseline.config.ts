import { defineConfig, mergeConfig } from '../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from '../../apps/tldw-frontend/vitest.config'
import path from 'node:path'
import fs from 'node:fs'
const source = path.resolve(__dirname, '../../apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx')
export default mergeConfig(base, defineConfig({
  root: path.resolve(__dirname, '../../apps/tldw-frontend'),
  plugins: [{ name: 'uat196-baseline-review', enforce: 'pre', load(id) {
    if (id.split('?')[0] === source) return fs.readFileSync(path.resolve(__dirname, '../uat196-repair-20260917/GeneratePanel.baseline.tsx'), 'utf8')
  } }]
}))
