import {defineConfig,mergeConfig} from '../../apps/tldw-frontend/node_modules/vitest/dist/config.js'
import base from '../../apps/tldw-frontend/vitest.config'
import path from 'node:path'
const config = mergeConfig(base,defineConfig({root:path.resolve(__dirname,'../../apps/tldw-frontend'),resolve:{alias:{'pa-tesseract.js':path.resolve(__dirname,'../../apps/node_modules/.bun/pa-tesseract.js@5.1.1/node_modules/pa-tesseract.js')}},test:{include:[path.resolve(__dirname,'*.test.{ts,tsx}')],maxWorkers:1}}))

config.test.include = [path.resolve(__dirname, "*.test.{ts,tsx}")]
export default config
