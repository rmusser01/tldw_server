import { createRequire } from 'node:module'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts'
const require=createRequire('/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/package.json')
export default {...base,plugins:[{name:'private-probe-package-resolution',enforce:'pre',resolveId(source,importer){if(source==='@tanstack/react-query')return require.resolve(source).replace(/index\.cjs$/, 'index.js');if(importer?.startsWith('/private/tmp/cycle4-uat111-retained-error-probe.test')&&!source.startsWith('.')&&!source.startsWith('/')&&!source.startsWith('@/')&&source!=='vitest'){try{return require.resolve(source)}catch{}}}}],test:{...base.test,include:['/private/tmp/cycle4-uat111-retained-error-probe.test.tsx']}}
