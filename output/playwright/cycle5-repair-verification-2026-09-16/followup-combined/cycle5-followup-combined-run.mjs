import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
const mode=process.argv[2],repo=process.cwd(),m=JSON.parse(fs.readFileSync('/private/tmp/cycle5-followup-combined-command.json'));
let cmd,args,cwd=repo,env={...process.env};
if(mode==='ui'||mode==='web'){cwd=path.join(repo,m[mode].cwd);cmd=path.join(cwd,'node_modules/.bin/vitest');args=['run',...m[mode].files,'--maxWorkers=1','--no-file-parallelism'];}
else if(mode==='typecheck'){cwd=path.join(repo,'apps/tldw-frontend');cmd=path.join(cwd,'node_modules/.bin/tsc');args=['--noEmit','--incremental','false'];env.NODE_OPTIONS='--max-old-space-size=8192';}
else throw Error('Unknown mode');
const log='/private/tmp/cycle5-followup-combined-'+mode+'.log',fd=fs.openSync(log,'w'),start=Date.now();
const child=spawn(cmd,args,{cwd,env,stdio:['ignore',fd,fd]});
child.on('exit',code=>{console.log(JSON.stringify({mode,code,elapsedMs:Date.now()-start,log}));process.exit(code??1)});
child.on('error',error=>{console.error(error);process.exit(1)});
