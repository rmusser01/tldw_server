import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts";
import fs from 'node:fs';
const sourceTarget="/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/db/dexie/server-chat-mirror.ts";
const testTarget="/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/db/dexie/__tests__/server-chat-chronology.test.ts";
export default {...base, plugins:[...(base.plugins||[]), {name:'uat156-extra-baseline', enforce:'pre',load(id){const file=id.split('?')[0]; if(file===sourceTarget) {process.stderr.write('UAT156_EXTRA_BASELINE_LOADED sha256=9474ea785c871a25954979efc07d34b344f3fbf09b53aec5348de63cbfbbc7c7\n');return fs.readFileSync('/private/tmp/uat156-independent-baseline-source.ts','utf8');} if(file===testTarget) return fs.readFileSync(testTarget,'utf8')+fs.readFileSync('/private/tmp/uat156-independent-extra-cases.ts','utf8'); }}]};
