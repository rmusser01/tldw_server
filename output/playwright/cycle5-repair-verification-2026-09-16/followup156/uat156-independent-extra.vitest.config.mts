import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts";
import fs from 'node:fs';
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/db/dexie/__tests__/server-chat-chronology.test.ts";
export default {...base, plugins:[...(base.plugins || []), {name:'uat156-independent-extra-cases', enforce:'pre', load(id) { if(id.split('?')[0] !== target) return; return fs.readFileSync(target,'utf8') + fs.readFileSync('/private/tmp/uat156-independent-extra-cases.ts','utf8'); } }]};
