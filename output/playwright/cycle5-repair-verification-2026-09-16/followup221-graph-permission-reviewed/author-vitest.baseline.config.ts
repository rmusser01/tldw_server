import { defineConfig, mergeConfig } from '../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from '../../apps/tldw-frontend/vitest.config'
import path from 'node:path'
import fs from 'node:fs'
const root = path.resolve(__dirname, '../..')
const files = new Map([
  ['components/Notes/NotesGraphWorkspace.tsx', 'NotesGraphWorkspace.tsx'],
  ['components/Notes/hooks/useNotesGraphWorkspace.tsx', 'useNotesGraphWorkspace.tsx'],
  ['assets/locale/en/option.json', 'option.json']
].map(([source, snapshot]) => [path.join(root, 'apps/packages/ui/src', source), path.join(__dirname, 'baseline', snapshot)]))
export default mergeConfig(base, defineConfig({
  root: path.join(root, 'apps/tldw-frontend'),
  plugins: [{ name: 'read-only-uat221-baseline', enforce: 'pre', load(id) {
    const snapshot = files.get(id.split('?')[0])
    return snapshot ? fs.readFileSync(snapshot, 'utf8') : null
  }}]
}))
