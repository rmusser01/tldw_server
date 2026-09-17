import fs from 'node:fs';
import path from 'node:path';
import base from '../../../apps/tldw-frontend/vitest.config';
const root=path.resolve(__dirname,'../../..');
const original=path.join(root,'apps/packages/ui/src/utils/chat-error-message.ts');
export default {...base,cacheDir:path.join(__dirname,'vite-baseline-cache'),plugins:[...(base.plugins as any[]),{
  name:'independent-model232-baseline',enforce:'pre',
  transform(code:string,id:string){return id.split('?')[0]===original?{code:fs.readFileSync(path.join(__dirname,'baseline-chat-error-message.ts'),'utf8'),map:null}:undefined;}
}]};
