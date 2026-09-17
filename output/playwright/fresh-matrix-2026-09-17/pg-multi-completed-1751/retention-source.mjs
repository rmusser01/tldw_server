import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(root,'../..');
const destination=path.join(repo,'output/playwright/fresh-matrix-2026-09-17/pg-multi-completed-1751');
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
for (const cell of ['sqlite-multi','pg-multi']) { const file=path.join(root,`fresh-final-20260917-${cell}.profile.private.json`); if(fs.existsSync(file)){const creds=read(read(file).credentialsPath);values.push(creds.apiKey,creds.jwtSecret,creds.apiHashSecret,...Object.values(creds.accounts).map(a=>a.password));}}
const secrets=[...new Set(values.filter(v=>typeof v==='string'&&v.length>0).flatMap(v=>[v,encodeURIComponent(v),JSON.stringify(v).slice(1,-1)]))];
const pgMultiRuntime=read(path.join(root,'holders/fresh-final-20260917-pg-multi/runtime.pg-config.private.json'));
secrets.push(...[pgMultiRuntime.password,encodeURIComponent(pgMultiRuntime.password),JSON.stringify(pgMultiRuntime.password).slice(1,-1)]);
const relative=['matrix-progress.json','pg-multi-api-isolation.mjs'];
for(const dir of ['native/pg-multi','audits']){
  for(const name of fs.readdirSync(path.join(root,dir)).sort()){
    if((dir==='audits'&&!name.startsWith('pg-multi-')&&!name.startsWith('testbot-pg-multi-'))||/private|credentials/i.test(name)||!(/\.(txt|json|md|js)$/.test(name)))continue;
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
const metadata={at:new Date().toISOString(),task:'TASK13260',frozenRevision:gate.revision,runId:gate.runId,status:'All48frozenmatrixoutcomes complete with failures;16newfindings231-246 unresolved',baseCheckpoint:'../sqlite-single-checkpoint-1331/manifest.json',credentialScan:{knownValues:secrets.length,candidateFiles:blobs.length,matches:0,jwtMatches:0,privateFilesExcluded:true},files:manifest};
fs.writeFileSync(path.join(destination,'manifest.json'),JSON.stringify(metadata,null,2)+'\n');
fs.copyFileSync(fileURLToPath(import.meta.url),path.join(destination,'retention-source.mjs'));
fs.writeFileSync(path.join(destination,'README.md'),`# Fresh matrix: PostgreSQL multi-user checkpoint

Frozen source: ${gate.revision}. All12 PostgreSQL multi-user rows now have bounded outcomes. All48 matrix outcomes are recorded across four separate cells, including failures and blocked dependencies. This is not full UAT acceptance.16new findings231–246 remain unresolved before any subsequent full matrix.

Includes native UI evidence, failed harness attempts, independent password-login API corroboration, sanitized diagnosis and source hashes. Private credentials/configurations/logs/input helpers excluded. All four profile credentials, provider secrets, fixture and both runtime PostgreSQL passwords plus JWT shapes are scanned. Larger100KB payloads are gzipped with verified roundtrip. The15frozen harness files remain unchanged.

Passes include normal setup/operatorprovider/admin UI accounts, ordinary Chat/retry/canonical7, actual image attachment under unsupported-modelguard, natural expiry1829.414s without open pages/workers or synthetic time, Pirate instruction/reload, Note/card/backlink, real outage/nativeRetry, and bounded reciprocal ownership. Pirate provider weather claims are unverified, not accuracy acceptance.

Failures:234generation browser30s timeout,235stale re-ratepreview,239world-book500,240Hardmisclassifiedlapse,242singularcopy,243ordinaryChatmodepresentation,245missingPGquotaschema blocksordinary/adminupload beforejobcreation,246TestBotstream45s timeout of unproven cause.231consolecopy and232errorfeedback also reproduce. ExactWikipediaexternallydenied. No persistedMedia means no sourceQA/reanalysis/Trash or populatedMedia/jobownership pass. No generated-five-card/mixed/earlyEnd, visioninference or truehidden acceptance.

NativeBack chains and directforeignNote links are verified bothdirections.30asserted APIchecks: ownNotesread/write200 with originalrestorationAlicev5/Bobv3, foreignNotesGET/validPUT404, Chatdetail/messages403, cards404 and onlyownpopulateddeck. Initialhelperexpected404forChat; actual403was safe and retained before correction. No browsertoken/storage extraction. Lateownershipcardswerenotstudied.

Ownedapps/browserstopped17:50UTC; portsrefused17:51. Officialfixtureholders and allisolateddata retained for repairs. See controlleranddiagnoses for exact IDs,times,limits.
`);
console.log(JSON.stringify({destination,files:manifest.length,bytes:manifest.reduce((n,x)=>n+x.bytes,0),manifestSha256:sha(fs.readFileSync(path.join(destination,'manifest.json'))),knownCredentialMatches:0}));
