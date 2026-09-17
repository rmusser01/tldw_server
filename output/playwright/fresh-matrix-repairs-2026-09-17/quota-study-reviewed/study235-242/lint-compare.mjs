import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
const require=createRequire(path.resolve('apps/tldw-frontend/package.json'));
const {ESLint}=require('eslint');
const files=['apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx',...['rerate','cram-mode','cram-completion'].map(name=>`apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.${name}.test.tsx`)];
const lint=new ESLint({cwd:process.cwd(),overrideConfigFile:path.resolve('apps/tldw-frontend/eslint.config.mjs')});
const results={};
for(const label of ['baseline','current']){
 const all=[];
 for(const file of files){const text=label==='baseline'?execFileSync('git',['show',`HEAD:${file}`],{encoding:'utf8'}):fs.readFileSync(file,'utf8');const [r]=await lint.lintText(text,{filePath:path.resolve(file)});all.push(...r.messages.map(m=>({file,rule:m.ruleId,message:m.message,severity:m.severity})));}
 results[label]=all;
}
const a=results.baseline.map(x=>JSON.stringify(x)).sort(),b=results.current.map(x=>JSON.stringify(x)).sort();
const r={baseline:results.baseline.length,current:results.current.length,exactSemanticMatch:JSON.stringify(a)===JSON.stringify(b),diagnostics:results.current};
fs.writeFileSync('.tmp/uat-repairs-231-246/study235-242/lint-comparison.json',JSON.stringify(r,null,2)+'\n');
console.log(JSON.stringify({baseline:r.baseline,current:r.current,exactSemanticMatch:r.exactSemanticMatch}));
