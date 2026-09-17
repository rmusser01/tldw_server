// Parent TASK13260: copy-only preparation after the recorded matrix gate opens.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const packet=path.dirname(fileURLToPath(import.meta.url)), repo=path.resolve(packet,'../..');
const [revision]=process.argv.slice(2), gateFile=path.join(packet,'release-gate.json');
if(!/^[0-9a-f]{40}$/.test(revision||'')||!fs.existsSync(gateFile))throw Error('An exact released commit and recorded gate are required');
const gate=JSON.parse(fs.readFileSync(gateFile));
if(gate.status!=='RELEASED'||gate.revision!==revision)throw Error('Gate does not match source revision');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const run=(cmd,args)=>{const r=spawnSync(cmd,args,{cwd:repo,encoding:'utf8',maxBuffer:4*1024*1024});if(r.status!==0)throw Error(`${cmd} failed (${r.status}): ${r.stderr}`);return r.stdout;};
const write=(file,value)=>{fs.mkdirSync(path.dirname(file),{recursive:true,mode:0o700});fs.writeFileSync(file,JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});};
const receipt=path.join(packet,'copy-preparation');fs.mkdirSync(receipt,{recursive:true,mode:0o700});
const tar=path.join(receipt,revision+'.tar');if(fs.existsSync(tar))throw Error('Preserve existing preparation; do not overwrite it');
run('git',['archive','--format=tar','-o',tar,revision]);
const tracked=run('git',['ls-tree','-r','-z','--name-only',revision]).split('\0').filter(Boolean);
const deps=path.join(packet,'dependencies/python-venv');if(fs.existsSync(deps))throw Error('Dependency copy already exists');fs.mkdirSync(path.dirname(deps),{recursive:true,mode:0o700});
const site='lib/python3.11/site-packages', originalSite=path.join(repo,'.venv',site);
const pths=fs.readdirSync(originalSite).filter(n=>n.endsWith('.pth')).sort();
const disabled=['__editable__.backlog_py-0.1.0.pth','__editable__.tldw_server-0.1.32.pth'];
if(JSON.stringify(pths.filter(n=>n.startsWith('__editable__')))!==JSON.stringify(disabled))throw Error('Editable inventory changed');
const before=pths.map(n=>({path:n,sha256:sha(fs.readFileSync(path.join(originalSite,n)))}));
const reviewedPth=JSON.parse(fs.readFileSync(path.join(packet,'reviewed-pth-inventory.json'))).files;
if(JSON.stringify(before)!==JSON.stringify(reviewedPth))throw Error('Reviewed full Python startup-hook inventory changed');
run('cp',['-ac',path.join(repo,'.venv'),deps]);
for(const n of disabled)fs.renameSync(path.join(deps,site,n),path.join(deps,site,n+'.disabled-uat'));
write(path.join(receipt,'python-dependency-copy.json'),{at:new Date().toISOString(),from:path.join(repo,'.venv'),to:deps,command:'cp -ac',reusedInstallation:true,disabledCopiedPthOnly:disabled,before,after:before.map(f=>({path:disabled.includes(f.path)?f.path+'.disabled-uat':f.path,sha256:sha(fs.readFileSync(path.join(deps,site,disabled.includes(f.path)?f.path+'.disabled-uat':f.path)))}))});
for(const cell of ['sqlite-single','sqlite-multi','pg-single','pg-multi']){
 const root=path.join(packet,'sources',cell);if(fs.existsSync(root))throw Error('Cell archive already exists: '+cell);fs.mkdirSync(root,{recursive:true,mode:0o700});
 run('tar',['-xf',tar,'-C',root]);
 const files=tracked.map(file=>{const p=path.join(root,file),st=fs.lstatSync(p);return st.isSymbolicLink()?{path:file,kind:'symlink',target:fs.readlinkSync(p)}:{path:file,kind:'file',bytes:st.size,sha256:sha(fs.readFileSync(p))};});
 write(path.join(receipt,cell+'-archive-manifest.json'),{at:new Date().toISOString(),revision,root,fileCount:files.length,files});
 for(const tree of ['apps/node_modules','apps/tldw-frontend/node_modules','apps/packages/ui/node_modules'])run('cp',['-ac',path.join(repo,tree),path.join(root,tree)]);
 const alias=path.join(root,'apps/node_modules/node_modules');if(!fs.lstatSync(alias).isSymbolicLink())throw Error('Unexpected self alias');fs.unlinkSync(alias);fs.symlinkSync('.',alias);
 const removed=[];for(const tree of ['apps/node_modules','apps/tldw-frontend/node_modules','apps/packages/ui/node_modules'])for(const cache of ['.vite','.cache']){const p=path.join(root,tree,cache);if(fs.existsSync(p)){fs.rmSync(p,{recursive:true,force:false});removed.push(path.relative(root,p));}}
 write(path.join(receipt,cell+'-dependency-copy.json'),{at:new Date().toISOString(),cell,command:'cp -ac',reusedInstallation:true,trees:['apps/node_modules','apps/tldw-frontend/node_modules','apps/packages/ui/node_modules'],copiedAliasTarget:'.',removedCopiedCaches:removed,noNextBuildCopied:true});
 console.log(JSON.stringify({cell,sourceFiles:files.length,root,dependenciesCopied:true}));
}
write(path.join(receipt,'complete.json'),{at:new Date().toISOString(),revision,gate:gateFile,archive:tar,archiveSha256:sha(fs.readFileSync(tar)),noRuntimeOrProfileStarted:true});
