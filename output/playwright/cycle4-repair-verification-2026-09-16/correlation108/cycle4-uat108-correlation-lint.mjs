import fs from 'node:fs';import cp from 'node:child_process';
const root=process.cwd(), {ui}=JSON.parse(fs.readFileSync('/private/tmp/cycle4-uat108-correlation-paths.json'));const base='7dcee3d72c';
const executable='apps/tldw-frontend/node_modules/.bin/eslint', config='apps/tldw-frontend/eslint.config.mjs';
const cur=cp.spawnSync(executable,['--config',config,'-f','json',...ui],{cwd:root,encoding:'utf8',maxBuffer:15e6});
if(!cur.stdout.startsWith('['))throw Error(cur.stderr);const current=JSON.parse(cur.stdout), baseline=[];
for(const p of ui){const source=cp.execFileSync('git',['show',base+':'+p],{encoding:'utf8',maxBuffer:15e6});const r=cp.spawnSync(executable,['--config',config,'-f','json','--stdin','--stdin-filename',p],{cwd:root,input:source,encoding:'utf8',maxBuffer:15e6});baseline.push(...JSON.parse(r.stdout));}
const flat=arr=>arr.flatMap(file=>file.messages.map(m=>({path:file.filePath.replace(root+'/',''),severity:m.severity,rule:m.ruleId,message:m.message})));
const b=flat(baseline),c=flat(current),diff=(a,b)=>{const pool=[...b].map(JSON.stringify);return a.filter(x=>{const i=pool.indexOf(JSON.stringify(x));if(i<0)return true;pool.splice(i,1);return false})};
const result={baseline:base,paths:ui,currentErrors:c.filter(x=>x.severity===2).length,currentWarnings:c.filter(x=>x.severity===1).length,baselineErrors:b.filter(x=>x.severity===2).length,baselineWarnings:b.filter(x=>x.severity===1).length,added:diff(c,b),removed:diff(b,c)};
for(const [suffix,value] of [['current',current],['baseline',baseline],['comparison',result]])fs.writeFileSync('/private/tmp/cycle4-uat108-correlation-lint-'+suffix+'.json',JSON.stringify(value,null,2)+'\n');console.log(JSON.stringify(result));
