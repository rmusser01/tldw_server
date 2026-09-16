import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts'
export default {...base,plugins:[{name:'independent-created-deck-removal',enforce:'pre',transform(code:string,id:string){if(!id.endsWith('FlashcardsManager.deck-authority.integration.test.tsx'))return;const at=code.lastIndexOf('\n})');return {code:code.slice(0,at)+'\n'+fs.readFileSync('/private/tmp/cycle5-uat151-created-deck-removal-probe.txt','utf8')+code.slice(at),map:null}}}]}
