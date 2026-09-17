import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const root=process.cwd(), dir=path.join(root,'.tmp/uat-repairs-231-246/review250');
const require=createRequire(path.join(root,'apps/tldw-frontend/package.json'));
const {ESLint}=require('eslint');
const eslint=new ESLint({cwd:root,overrideConfigFile:path.join(root,'apps/tldw-frontend/eslint.config.mjs')});
const file='apps/packages/ui/src/services/__tests__/background-proxy.test.ts';
for(const [label,text] of [['baseline',fs.readFileSync(path.join(root,'.tmp/uat-repairs-231-246/transport-fixture250/original.test.txt'),'utf8')],['current',fs.readFileSync(file,'utf8')]]){
 const result=await eslint.lintText(text,{filePath:path.join(root,file)});
 fs.writeFileSync(path.join(dir,`eslint-${label}.json`),JSON.stringify(result,null,2)+'\n');
 console.log(label,result.reduce((s,r)=>s+r.errorCount,0),result.reduce((s,r)=>s+r.warningCount,0));
}
