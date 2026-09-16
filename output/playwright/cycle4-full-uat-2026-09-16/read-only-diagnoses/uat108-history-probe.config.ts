import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/node_modules/.bun/vitest@4.0.18+08ee8852a9d25cb0/node_modules/vitest/dist/config.cjs'
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const target=path.join(root,'src/utils/__tests__/generate-history.image-generation.test.ts')
const request=JSON.parse(fs.readFileSync('/private/tmp/uat-cycle4-single-configured-chat-request.txt','utf8'))
const addition=`
import {decodeChatErrorPayload,buildAssistantErrorContent} from '@/utils/chat-error-message';
describe('UAT108 actual history serializer with captured retry input',()=>{
  const original = ${JSON.stringify(request.messages.slice(0,2))};
  it('excludes the valid displayed failure from the model history',()=>{
    expect(decodeChatErrorPayload(original[1].content)).not.toBeNull();
    const result=generateHistory(original as any,'gpt-4o-mini');
    console.log('UAT108_SERIALIZER',JSON.stringify({inputRoles:original.map(x=>x.role),outputRoles:result.map(x=>x._getType()),encodedFailureInModel:JSON.stringify(result).includes('__tldw_error__:')}));
    expect(result).toHaveLength(1);
  });
  it('preserves a user quotation and malformed assistant marker',()=>{
    const result=generateHistory([{role:'user',content:original[1].content},{role:'assistant',content:'__tldw_error__:not-json'}],'gpt-4o-mini');
    expect(result).toHaveLength(2);
    expect(JSON.stringify(result)).toContain('__tldw_error__:not-json');
  });
  it('preserves legitimate partial text produced before a transport failure',()=>{
    const partial=buildAssistantErrorContent('Partial legitimate answer',new Error('Transport failed'));
    expect(partial).toBe('Partial legitimate answer');
    const result=generateHistory([{role:'assistant',content:partial}],'gpt-4o-mini');
    expect(result).toHaveLength(1);
    expect(JSON.stringify(result)).toContain('Partial legitimate answer');
  });
});`
export default defineConfig({root,plugins:[{name:'private-uat108-probe',enforce:'pre',load(id){if(id.split('?')[0]===target)return fs.readFileSync(target,'utf8')+addition}}],resolve:{alias:{'@':path.join(root,'src'),'~':path.join(root,'src')}},test:{environment:'jsdom',setupFiles:[path.join(root,'vitest.setup.ts')],include:[target],maxWorkers:1,fileParallelism:false,restoreMocks:true}})
