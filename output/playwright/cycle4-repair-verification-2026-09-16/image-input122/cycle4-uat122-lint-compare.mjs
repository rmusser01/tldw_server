import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {ESLint} from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/node_modules/eslint/lib/api.js';
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2';
const current=JSON.parse(fs.readFileSync('/private/tmp/cycle4-uat122-eslint-final.json','utf8'));
const eslint=new ESLint({cwd:root,overrideConfigFile:path.join(root,'apps/tldw-frontend/eslint.config.mjs')});
const baseline=[];
const baseRef=JSON.parse(fs.readFileSync('/private/tmp/cycle4-uat122-owned-manifest.json','utf8')).baseRef;
for(const item of current){const relative=path.relative(root,item.filePath);let source;try{source=execFileSync('git',['show',`${baseRef}:${relative}`],{cwd:root,encoding:'utf8',stdio:['ignore','pipe','ignore']});}catch{continue;}baseline.push(...await eslint.lintText(source,{filePath:item.filePath}));}
const counts=items=>{let result={};for(const item of items)for(const m of item.messages){const k=JSON.stringify([path.relative(root,item.filePath),m.ruleId,m.severity,m.message]);result[k]=(result[k]||0)+1;}return result;};
const old=counts(baseline),now=counts(current);const added=[],removed=[];for(const k of new Set([...Object.keys(old),...Object.keys(now)])){const n=(now[k]||0)-(old[k]||0);if(n>0)added.push({diagnostic:JSON.parse(k),count:n});else if(n<0)removed.push({diagnostic:JSON.parse(k),count:-n});}
const summary=items=>({errors:items.reduce((n,i)=>n+i.errorCount,0),warnings:items.reduce((n,i)=>n+i.warningCount,0),files:items.length});
fs.writeFileSync('/private/tmp/cycle4-uat122-eslint-baseline.json',JSON.stringify(baseline,null,2)+'\n');
fs.writeFileSync('/private/tmp/cycle4-uat122-eslint-comparison.json',JSON.stringify({baselineRef:baseRef,baseline:summary(baseline),current:summary(current),added,removed},null,2)+'\n');
