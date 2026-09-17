import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
const root=process.cwd();
const require=createRequire(path.join(root,'apps/tldw-frontend/package.json'));
const { ESLint }=require('eslint');
const out='.tmp/uat-repairs-231-246/review-media/';
const files=JSON.parse(fs.readFileSync('.tmp/uat-repairs-231-246/media237-241-244/paths.json','utf8'));
const lint=new ESLint({cwd:root,overrideConfigFile:path.join(root,'apps/tldw-frontend/eslint.config.mjs')});
for(const revision of ['baseline','current']) {
 const results=[];
 for(const file of files) {
  const text=revision==='baseline'?execFileSync('git',['show',`HEAD:${file}`],{encoding:'utf8'}):fs.readFileSync(file,'utf8');
  results.push(...await lint.lintText(text,{filePath:file}));
 }
 fs.writeFileSync(out+'eslint-'+revision+'.json',JSON.stringify(results,null,2)+'\n');
 console.log(revision,results.reduce((n,x)=>n+x.errorCount,0),results.reduce((n,x)=>n+x.warningCount,0));
}
