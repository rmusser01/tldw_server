import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts";
import { execFileSync } from "node:child_process";
const root="/Users/macbook-dev/Documents/GitHub/tldw_server2";
const files=new Set(["hooks/chat-modes/chatModePipeline.ts","hooks/chat-helper/index.ts","models/index.ts","models/ChatTldw.ts","services/tldw/TldwChat.ts"].map(p=>root+"/apps/packages/ui/src/"+p));
export default {...base,plugins:[...(base.plugins || []),{name:"uat108-unchanged-source-replay",enforce:"pre",load(id){if(files.has(id.split("?")[0]))return execFileSync("git",["show","8097d672d5:"+id.slice(root.length+1)],{cwd:root,encoding:"utf8"});}}],test:{...base.test,root:root+"/apps/packages/ui",include:["src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx"],testNamePattern:"real failed-turn regeneration",maxWorkers:1}};
