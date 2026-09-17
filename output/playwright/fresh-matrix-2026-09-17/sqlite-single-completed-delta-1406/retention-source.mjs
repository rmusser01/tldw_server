import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(root,'../..');
const destination=path.join(repo,'output/playwright/fresh-matrix-2026-09-17/sqlite-single-completed-delta-1406');
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
const secrets=[...new Set(values.filter(v=>typeof v==='string'&&v.length>0).flatMap(v=>[v,encodeURIComponent(v),JSON.stringify(v).slice(1,-1)]))];
const relative=[...gate.harness.map(f=>f.path),'release-gate.json','matrix-progress.json'];
for(const dir of ['native/sqlite-single','audits','fixtures','copy-preparation']){
  for(const name of fs.readdirSync(path.join(root,dir)).sort()){
    if(/private|credentials/i.test(name)||!(/\.(txt|json|md|js)$/.test(name)))continue;
    const rel=path.join(dir,name);
    if(!fs.lstatSync(path.join(root,rel)).isFile())continue;
    relative.push(rel);
  }
}
const base=read(path.join(repo,'output/playwright/fresh-matrix-2026-09-17/sqlite-single-checkpoint-1331/manifest.json'));
const previous=new Map(base.files.map(f=>[f.path.replace(/\.gz$/,''),f.sourceSha256]));
const candidates=relative.filter(rel=>sha(fs.readFileSync(path.join(root,rel)))!==previous.get(rel)).map(rel=>({source:path.join(root,rel),rel}));
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
const metadata={at:new Date().toISOString(),task:'TASK13260',frozenRevision:gate.revision,runId:gate.runId,status:'Completed SQLite-single pass with failures; other cells pending',baseCheckpoint:'../sqlite-single-checkpoint-1331/manifest.json',credentialScan:{knownValues:secrets.length,candidateFiles:blobs.length,matches:0,jwtMatches:0,privateFilesExcluded:true},files:manifest};
fs.writeFileSync(path.join(destination,'manifest.json'),JSON.stringify(metadata,null,2)+'\n');
fs.copyFileSync(fileURLToPath(import.meta.url),path.join(destination,'retention-source.mjs'));
fs.writeFileSync(path.join(destination,'README.md'),`# Fresh matrix: SQLite single-user checkpoint\n\nFrozen source: ${gate.revision}. This is partial evidence, not full UAT signoff. Other three configurations remain pending. See controller/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md for per-row outcomes and limits. New issues231–237 remain unresolved.\n\nThis delta retains new or changed native browser output, failures, independent audits and controller state since sqlite-single-checkpoint-1331. The base packet retains unchanged harness and preparation. Source archives and dependencies are not copied. Payloads above100KB are losslessly gzipped and roundtrip-verified. Known profile/provider/PostgreSQL credentials and JWT shapes were checked before writing; private files, private logs and credential-entry helper are excluded.\n\nPrompt execution uses real local inference. Text-model image guard evidence does not certify vision. Re-rate adds an intentional third scheduled review but displays a stale interval (235). Biology generation fails at the30-second frontend proxy (234), blocking exact-five-card and mixed-deck study.\n\nOriginal working paths inside evidence preserve provenance; corresponding retained files use the same relative paths within this packet.\n`);
console.log(JSON.stringify({destination,files:manifest.length,bytes:manifest.reduce((n,x)=>n+x.bytes,0),manifestSha256:sha(fs.readFileSync(path.join(destination,'manifest.json'))),knownCredentialMatches:0}));
