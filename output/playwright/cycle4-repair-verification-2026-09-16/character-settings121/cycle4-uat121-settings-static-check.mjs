import fs from 'node:fs'; import cp from 'node:child_process'; import path from 'node:path';
const root=process.cwd();
const paths=JSON.parse(fs.readFileSync('/private/tmp/cycle4-uat121-settings-paths.json','utf8'));
fs.writeFileSync('/private/tmp/cycle4-uat121-settings-frontend-paths.json',JSON.stringify(paths,null,2)+'\n');
const base=[];
for(const file of paths){ const old=cp.spawnSync('git',['show',`HEAD:${file}`],{encoding:'utf8'}); if(old.status!==0) continue; const input=old.stdout; const res=cp.spawnSync(path.join(root,'apps/tldw-frontend/node_modules/.bin/eslint'),['--config','apps/tldw-frontend/eslint.config.mjs','--stdin','--stdin-filename',file,'--format','json'],{input,encoding:'utf8'}); try{base.push(...JSON.parse(res.stdout))}catch{throw Error(res.stderr)} }
fs.writeFileSync('/private/tmp/cycle4-uat121-settings-lint-baseline.json',JSON.stringify(base,null,2)+'\n');
const res=cp.spawnSync(path.join(root,'apps/tldw-frontend/node_modules/.bin/eslint'),['--config','apps/tldw-frontend/eslint.config.mjs',...paths,'--format','json'],{encoding:'utf8',maxBuffer:16*1024*1024});
const current=JSON.parse(res.stdout); fs.writeFileSync('/private/tmp/cycle4-uat121-settings-lint-current.json',JSON.stringify(current,null,2)+'\n');
const signatures=rows=>rows.flatMap(r=>r.messages.map(m=>JSON.stringify([path.relative(root,r.filePath),m.ruleId,m.severity,m.message.replace(/at line \d+/g,"at line #")]))).sort();
const old=signatures(base),now=signatures(current),remain=[...old],added=[];
for(const s of now){let i=remain.indexOf(s);if(i<0)added.push(s);else remain.splice(i,1)}
const report={baseline:old.length,current:now.length,errors:current.reduce((n,r)=>n+r.errorCount,0),warnings:current.reduce((n,r)=>n+r.warningCount,0),added,removed:remain,coveredFiles:current.length};
fs.writeFileSync('/private/tmp/cycle4-uat121-settings-lint-comparison.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
