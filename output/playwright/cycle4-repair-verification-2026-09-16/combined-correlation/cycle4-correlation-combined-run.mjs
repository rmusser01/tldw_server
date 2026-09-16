import fs from 'node:fs';import path from 'node:path';import {spawn} from 'node:child_process';
const mode=process.argv[2],repo=process.cwd(),m=JSON.parse(fs.readFileSync('/private/tmp/cycle4-correlation-combined-command.json'));let cmd,args,cwd=repo;let env={...process.env};
if(mode==='ui'){cwd=path.join(repo,m.ui.cwd);cmd=path.join(cwd,'node_modules/.bin/vitest');args=['run',...m.ui.files,'--maxWorkers=1','--no-file-parallelism'];}
else if(mode==='backend'){cmd=path.join(repo,'.venv/bin/python');args=['-m','pytest',...m.backend.files,'-q'];}
else if(mode==='bandit'){cmd=path.join(repo,'.venv/bin/python');args=['-m','bandit',...m.bandit.files,'-f','json','-o','/private/tmp/cycle4-correlation-combined-bandit.json'];}
else if(mode==='typecheck'){cwd=path.join(repo,'apps/tldw-frontend');cmd=path.join(cwd,'node_modules/.bin/tsc');args=['--noEmit','--incremental','false'];env.NODE_OPTIONS='--max-old-space-size=8192';}
else throw new Error('Unknown verification mode');
const log='/private/tmp/cycle4-correlation-combined-'+mode+'.log',fd=fs.openSync(log,'w');const start=Date.now();const child=spawn(cmd,args,{cwd,env,stdio:['ignore',fd,fd]});child.on('exit',code=>{console.log(JSON.stringify({mode,code,elapsedMs:Date.now()-start,log}));process.exit(code??1);});child.on('error',e=>{console.error(e);process.exit(1);});
