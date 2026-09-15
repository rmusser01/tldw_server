import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config.ts";
import fs from 'node:fs';
const originals=JSON.parse(fs.readFileSync('/private/tmp/uat034-api-baseline.json','utf8'));
export default {...base, plugins:[{name:'uat034-baseline',enforce:'pre',load(id){return originals[id]}},...base.plugins]};
