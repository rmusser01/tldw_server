// TASK13260.200 / UAT258: independent retention-only verification; no native activity.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root = process.cwd();
const dir = '.tmp/uat-repairs-231-246/capability258-native-diagnosis';
const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/native-capability258-accepted';
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const checks = [], observations = [];
const check = (name, pass) => checks.push({name, pass: Boolean(pass)});
const read = p => fs.readFileSync(path.resolve(root, p));
const contained = (p, base=root) => {
  const absolute=path.resolve(base,p), relative=path.relative(base,absolute);
  if(relative==='..'||relative.startsWith('../')||path.isAbsolute(relative)) return false;
  let cursor=base;
  for(const part of relative.split(path.sep).filter(Boolean)) {
    cursor=path.join(cursor,part);
    if(fs.lstatSync(cursor).isSymbolicLink()) return false;
  }
  return true;
};
const walk = p => fs.readdirSync(p,{withFileTypes:true}).flatMap(e => {
  if(e.isSymbolicLink()) throw Error('Packet symlink rejected');
  const next=path.join(p,e.name);
  return e.isDirectory()?walk(next):[path.relative(path.resolve(packet),path.resolve(next))];
});
const inventory=walk(packet).sort();
const manifestBytes=read(packet+'/manifest.json');
const manifest=JSON.parse(manifestBytes);
const original=JSON.parse(read(dir+'/audit.json'));
const expectedManifest='5f6335979143612103e6cd0c5445fb3decf62edd18e9a91ff1e9721df2bb9702';
check('Exact requested manifest hash',sha(manifestBytes)===expectedManifest);
check('Only UAT258 accepted on reviewed revision; full matrix explicitly false',JSON.stringify(manifest.acceptedFindings)==='[258]'&&manifest.revision==='edfd06ec40a173f2e38ec65af715abb29f3aa002'&&manifest.fullMatrixAccepted===false);
check('Original bounded native audit remains CLEAR with 18 passing checks and 53 inputs',original.verdict.startsWith('CLEAR bounded258')&&original.checks.length===18&&original.checks.every(c=>c.pass)&&original.inputs.length===53);
const expectedNames=['REVIEW.md','audit.mjs','audit.json'].map(n=>dir+'/'+n);
check('Three expected identity payloads only',manifest.files.length===3&&manifest.files.every(e=>expectedNames.includes(e.source)&&e.path==='evidence/'+e.source&&e.encoding==='identity'));
const expectedInventory=['README.md','manifest.json','CHECKPOINT_SHA256SUMS',...manifest.files.map(e=>e.path)].sort();
check('Exact six-file inventory with no symlinks or path escape',JSON.stringify(inventory)===JSON.stringify(expectedInventory)&&inventory.every(p=>contained(p,path.resolve(packet))));
for(const e of manifest.files) {
  const b=read(packet+'/'+e.path),s=read(e.source);
  observations.push({kind:'payload',path:e.path,source:e.source,bytes:b.length,sha256:sha(b),sourceBytes:s.length,sourceSha256:sha(s)});
  check('Payload/source/manifest exact byte parity: '+path.basename(e.source),contained(e.source)&&b.equals(s)&&b.length===e.bytes&&s.length===e.sourceBytes&&sha(b)===e.sha256&&sha(s)===e.sourceSha256);
}
const sums=read(packet+'/CHECKPOINT_SHA256SUMS').toString().trim().split('\n').map(line=>{const m=/^([a-f0-9]{64})  (.+)$/.exec(line);if(!m)throw Error('Malformed checksum line');return{sha256:m[1],path:m[2]};});
check('Checksum coverage includes every packet file except checksum itself exactly once',new Set(sums.map(e=>e.path)).size===sums.length&&JSON.stringify(sums.map(e=>e.path).sort())===JSON.stringify(inventory.filter(p=>p!=='CHECKPOINT_SHA256SUMS')));
check('All checksum hashes match contained stored bytes',sums.every(e=>contained(e.path,path.resolve(packet))&&sha(read(packet+'/'+e.path))===e.sha256));
const byPath=new Map(manifest.omitted.map(e=>[e.path,e]));
check('Every original input represented once with original byte/hash/private metadata',manifest.omitted.length===53&&byPath.size===53&&original.inputs.every(e=>{const o=byPath.get(e.path);return o&&o.bytes===e.bytes&&o.sha256===e.sha256&&o.privateHashOnly===e.privateHashOnly&&typeof o.reason==='string'&&o.reason.length>0;}));
let prefixCount=0, exactCount=0, validInputs=true;
for(const e of original.inputs) {
  const b=read(e.path),hash=sha(b),o=byPath.get(e.path);
  const exact=hash===e.sha256&&b.length===e.bytes;
  const prefix=!exact&&e.path.endsWith('.private.log')&&e.privateHashOnly===true&&b.length>e.bytes&&sha(b.subarray(0,e.bytes))===e.sha256&&o.retentionVerification==='Original reviewed log prefix remains byte-identical; live log appended after audit';
  validInputs=validInputs&&contained(e.path)&&Boolean(exact||prefix);
  if(exact)exactCount++;if(prefix)prefixCount++;
  observations.push({kind:'original-input',path:e.path,observedBytes:e.bytes,observedSha256:e.sha256,currentBytes:b.length,currentSha256:hash,privateHashOnly:e.privateHashOnly,verification:exact?'exact':prefix?'original-prefix-identical-live-log-appended':'MISMATCH'});
}
check('All 53 local input hashes match exactly or disclosed live-log prefix',validInputs);
check('Only the disclosed private live log uses prefix proof',prefixCount===1&&exactCount===52&&manifest.omitted.filter(e=>e.retentionVerification).length===1);
check('All 13 private inputs omitted as raw payloads',original.inputs.filter(e=>e.privateHashOnly).length===13&&original.inputs.filter(e=>e.privateHashOnly).every(e=>!manifest.files.some(f=>f.source===e.path))&&manifest.files.every(e=>!/(?:private|\.log$|\.txt$)/i.test(e.source)));
const readme=read(packet+'/README.md').toString();
check('README preserves observer correction, prefix caveat and no standalone replay',readme.includes('old general observer did not match hyphenated ingestion-sources')&&readme.includes('not a zero-dispatch proof')&&readme.includes('original reviewed byte prefix')&&readme.includes('not standalone replay'));
check('README leaves UAT264 open and disclaims Sources/full-UAT acceptance',readme.includes('Separate UAT264 remains open')&&readme.includes('not Sources functionality or full UAT'));
const secrets=[];
const matrix='.tmp/uat-next-matrix-20260916';
const profiles=fs.readdirSync(matrix).filter(n=>n.endsWith('.profile.private.json'));
for(const n of profiles) {
  const p=JSON.parse(read(matrix+'/'+n)),c=JSON.parse(read(p.credentialsPath));
  secrets.push(c.apiKey,c.jwtSecret,c.apiHashSecret,...Object.values(c.accounts||{}).map(a=>a.password),...(c.providerSecrets||[]));
  if(p.pgConfigPath)secrets.push(JSON.parse(read(p.pgConfigPath)).password);
}
secrets.push(JSON.parse(read('.tmp/fresh-uat-recovery-20260916/postgres-private.json')).password);
const variants=[...new Set(secrets.filter(v=>typeof v==='string'&&v.length).flatMap(v=>[v,encodeURIComponent(v),JSON.stringify(v).slice(1,-1),Buffer.from(v).toString('base64'),Buffer.from(v).toString('base64url')]))];
let matches=0,jwtMatches=0;
for(const p of inventory) {const s=read(packet+'/'+p).toString();matches+=variants.filter(v=>s.includes(v)).length;jwtMatches+=(s.match(/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g)||[]).length;}
check('Independent known-credential and JWT scan over all six files finds zero candidates',matches===0&&jwtMatches===0&&variants.length===manifest.secretScan.knownVariants&&manifest.secretScan.matches===0&&manifest.secretScan.jwtMatches===0);
const retainer='.tmp/uat-repairs-231-246/retain-native-capability258.mjs';
const retainerText=read(retainer).toString();
check('Retainer refuses overwrite and changed non-prefix inputs',retainerText.includes('Refusing to overwrite retained packet')&&retainerText.includes("if(actual!==e.sha256&&!prefix)throw Error('Reviewed input changed: '"));
for(const p of [packet+'/README.md',packet+'/manifest.json',packet+'/CHECKPOINT_SHA256SUMS',retainer,dir+'/retention-audit.mjs']) {const b=read(p);observations.push({kind:'review-input',path:p,bytes:b.length,sha256:sha(b)});}
const result={task:'UAT258 native packet retention review',at:new Date().toISOString(),verdict:checks.every(c=>c.pass)?'CLEAR bounded UAT258 retention':'GAPS',checks,summary:{payloads:3,totalFiles:6,omittedReferences:53,privateOmittedReferences:13,exactCurrentInputs:exactCount,appendedLiveLogPrefixInputs:prefixCount,manifestSha256:sha(manifestBytes),credentialScan:{profiles:profiles.length,knownVariants:variants.length,files:inventory.length,matches,jwtMatches}},limits:['Acceptance remains UAT258 capability gating only; UAT264 Sources redirect remains open.','Retained safe reports are byte-identical; 53 original input references are hash-only, including one script also present as a safe payload. Omitted native/private/source inputs are not standalone replay.','The live log prefix matches its original observed length/hash. Later appended bytes are neither original evidence nor part of the retained payload. Current whole-log hash is an observation only.','Known-credential/JWT scanning is bounded to available local credentials and those encodings, not a general guarantee of absence of every possible secret.','No new native actions, tests, model requests, runtime/DB/Git/Backlog/product changes; only these retention review artifacts were written.','Retainer header mentions old task IDs; this is stale administrative commentary, while manifest and packet acceptance consistently identify UAT258.'],observations};
fs.writeFileSync(dir+'/retention-audit.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({verdict:result.verdict,checks:checks.length,failed:checks.filter(c=>!c.pass).map(c=>c.name),summary:result.summary}));
