import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts";
import fs from 'node:fs';
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/db/dexie/server-chat-mirror.ts";
export default {...base, plugins: [...(base.plugins || []), { name:'uat156-review-baseline', enforce:'pre', load(id) { if (id.split('?')[0] !== target) return; process.stderr.write('UAT156_BASELINE_LOADED sha256=9474ea785c871a25954979efc07d34b344f3fbf09b53aec5348de63cbfbbc7c7 source=2d5ad06c86cf279fe0fcc6445009813d97ab1c1b\n'); return fs.readFileSync('/private/tmp/uat156-independent-baseline-source.ts','utf8'); } }]};
