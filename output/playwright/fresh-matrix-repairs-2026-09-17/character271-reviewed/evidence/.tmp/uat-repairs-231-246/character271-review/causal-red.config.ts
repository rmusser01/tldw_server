import fs from 'node:fs'
import path from 'node:path'
import base from '../../../apps/packages/ui/vitest.config'
const hook = path.resolve(__dirname, '../../../apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx')
const baseline = fs.readFileSync(path.resolve(__dirname, 'baseline-useCharacterCrud.tsx'), 'utf8')
export default {
  ...base,
  plugins: [{ name: 'review-baseline-hook-only', enforce: 'pre' as const, load(id: string) { if (id.split('?')[0] === hook) return baseline } }],
}
