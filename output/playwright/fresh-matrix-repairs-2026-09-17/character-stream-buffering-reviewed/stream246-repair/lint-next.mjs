import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const root=process.cwd(), dir=path.join(root,'.tmp/uat-repairs-231-246/stream246-repair');
const require=createRequire(path.join(root,'apps/tldw-frontend/package.json'));
const {ESLint}=require('eslint');
const eslint=new ESLint({cwd:root,overrideConfigFile:path.join(root,'apps/tldw-frontend/eslint.config.mjs')});
const file='apps/tldw-frontend/scripts/__tests__/quickstart-proxy-timeout.test.mjs';
for(const [label,text] of [['baseline',fs.readFileSync(path.join(dir,'original-next-test.txt'),'utf8')],['current',fs.readFileSync(file,'utf8')]]){
 const result=await eslint.lintText(text,{filePath:path.join(root,file)});
 fs.writeFileSync(path.join(dir,`eslint-${label}.json`),JSON.stringify(result,null,2)+'\n');
 console.log(label,result.reduce((s,r)=>s+r.errorCount,0),result.reduce((s,r)=>s+r.warningCount,0));
}
