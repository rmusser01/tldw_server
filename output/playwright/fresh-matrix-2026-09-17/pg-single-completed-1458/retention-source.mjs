import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(root,'../..');
const destination=path.join(repo,'output/playwright/fresh-matrix-2026-09-17/pg-single-completed-1458');
if(fs.existsSync(destination))throw Error('Checkpoint already exists');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const gate=read(path.join(root,'release-gate.json'));
for(const f of gate.harness)if(sha(fs.readFileSync(path.join(root,f.path)))!==f.sha256)throw Error('Frozen harness changed: '+f.path);
const p=read(path.join(root,'fresh-final-20260917-sqlite-single.profile.private.json'));
const c=read(p.credentialsPath);
const pg=read(path.join(repo,'.tmp/fresh-uat-recovery-20260916/postgres-private.json'));
const values=[c.apiKey,c.jwtSecret,c.apiHashSecret,...Object.values(c.accounts).map(a=>a.password),...(c.providerSecrets||[]),pg.password];
const pgRuntime=read(path.join(root,'holders/fresh-final-20260917-pg-single/runtime.pg-config.private.json'));
const pgProfile=read(path.join(root,'fresh-final-20260917-pg-single.profile.private.json'));
const pgCredentials=read(pgProfile.credentialsPath);
values.push(pgRuntime.password,pgCredentials.apiKey,pgCredentials.jwtSecret,pgCredentials.apiHashSecret,...Object.values(pgCredentials.accounts).map(a=>a.password));
for (const cell of ['sqlite-multi']) { const file=path.join(root,`fresh-final-20260917-${cell}.profile.private.json`); if(fs.existsSync(file)){const creds=read(read(file).credentialsPath);values.push(creds.apiKey,creds.jwtSecret,creds.apiHashSecret,...Object.values(creds.accounts).map(a=>a.password));}}
const secrets=[...new Set(values.filter(v=>typeof v==='string'&&v.length>0).flatMap(v=>[v,encodeURIComponent(v),JSON.stringify(v).slice(1,-1)]))];
const relative=['matrix-progress.json'];
for(const dir of ['native/pg-single','audits']){
  for(const name of fs.readdirSync(path.join(root,dir)).sort()){
    if((dir==='audits'&&!name.startsWith('pg-single-'))||/private|credentials/i.test(name)||!(/\.(txt|json|md|js)$/.test(name)))continue;
    const rel=path.join(dir,name);
    if(!fs.lstatSync(path.join(root,rel)).isFile())continue;
    relative.push(rel);
  }
}
const base=read(path.join(repo,'output/playwright/fresh-matrix-2026-09-17/sqlite-single-checkpoint-1331/manifest.json'));
const previous=new Map(base.files.map(f=>[f.path.replace(/\.gz$/,''),f.sourceSha256]));
const candidates=relative.map(rel=>({source:path.join(root,rel),rel}));
candidates.push({source:path.join(repo,'Docs/Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md'),rel:'controller/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md'});
const blobs=[];
for(const {source,rel} of candidates){
  const raw=fs.readFileSync(source), text=raw.toString('utf8');
  if(secrets.some(s=>text.includes(s)))throw Error('Known credential in candidate: '+rel);
  if(/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/.test(text))throw Error('JWT-shaped credential in candidate: '+rel);
  const compressed=raw.length>100000;
  blobs.push({source,rel:rel+(compressed?'.gz':''),raw,bytes:compressed?zlib.gzipSync(raw):raw,compressed});
}
fs.mkdirSync(destination,{recursive:true});
const manifest=[];
for(const b of blobs){
  const out=path.join(destination,b.rel);fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,b.bytes);
  if(sha(fs.readFileSync(b.source))!==sha(b.raw))throw Error('Source changed during retention: '+b.rel);
  if(b.compressed&&sha(zlib.gunzipSync(fs.readFileSync(out)))!==sha(b.raw))throw Error('Compression roundtrip mismatch');
  manifest.push({path:b.rel,sha256:sha(b.bytes),bytes:b.bytes.length,sourceSha256:sha(b.raw),sourceBytes:b.raw.length,compression:b.compressed?'gzip':null});
}
const metadata={at:new Date().toISOString(),task:'TASK13260',frozenRevision:gate.revision,runId:gate.runId,status:'Completed PostgreSQL single-user pass with failures; both multi-user cells pending',baseCheckpoint:'../sqlite-single-checkpoint-1331/manifest.json',credentialScan:{knownValues:secrets.length,candidateFiles:blobs.length,matches:0,jwtMatches:0,privateFilesExcluded:true},files:manifest};
fs.writeFileSync(path.join(destination,'manifest.json'),JSON.stringify(metadata,null,2)+'\n');
fs.copyFileSync(fileURLToPath(import.meta.url),path.join(destination,'retention-source.mjs'));
fs.writeFileSync(path.join(destination,'README.md'),`# Fresh matrix: PostgreSQL single-user checkpoint\n\nFrozen source: ${gate.revision}. All12 single-user PostgreSQL rows have bounded outcomes, with failures; this is not full UAT acceptance. SQLite single-user evidence is in the two preceding packets. Fresh multi-user cells are pending.\n\nThis packet preserves public native evidence, harness errors, source diagnoses, independent reviews and controller state. It excludes private credentials/configs/logs and private input helpers. Known profile/provider/PostgreSQL values and JWT shapes were checked; payloads above100KB are gzipped and roundtrip-verified. Unchanged15 frozen harness and four archive/dependency preflights remain in the base SQLite packet.\n\nNew findings231–240 remain open. Media ingest238 fails restricted-role RLS; source-dependent QA/reanalysis/Trash cannot pass. Exact Wikipedia denied; five-card generation234 and Character236/239 fail. Re-rate235 shows stale preview and Hard analytics240 misclassifies recall. One actual Pirate answer exercises independent Note/card controls; this is not a grounded-source substitute. Image guard is not successful vision.\n\nBrowser closure interrupted the session after practice; fresh browser normal key entry recovered saved data. Later native Disconnect/reconnect and controlled same-profile API outage/Retry passed. Single-user JWT expiry is not applicable. Owned apps/browser stopped; official PG holder/data retained for repairs.\n\nSee controller matrix for exact scope and failures. Original working paths preserve provenance; retained files use matching relative paths.\n`);
console.log(JSON.stringify({destination,files:manifest.length,bytes:manifest.reduce((n,x)=>n+x.bytes,0),manifestSha256:sha(fs.readFileSync(path.join(destination,'manifest.json'))),knownCredentialMatches:0}));
