import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../../..');
const packet=path.join(root,'output/playwright/fresh-matrix-repairs-2026-09-17/native-config-ingest-auth-accepted');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const checks=[];
const check=(name,passed)=>checks.push({name,passed:Boolean(passed)});
const relative=p=>path.relative(root,path.resolve(root,p));
const safeRelative=p=>typeof p==='string'&&p.length>0&&!path.isAbsolute(p)&&!p.split(/[\\/]/).some(x=>x==='..'||x==='.'||x==='')&&!p.includes('\0');
function noSymlinkUnder(base,relativePath){
 if(!safeRelative(relativePath))return false;
 let current=base;
 for(const part of relativePath.split('/')){current=path.join(current,part);if(fs.lstatSync(current).isSymbolicLink())return false;}
 return true;
}
function walk(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>{
 const p=path.join(dir,e.name),s=fs.lstatSync(p);
 if(s.isSymbolicLink())return [{path:p,kind:'symlink'}];
 if(s.isDirectory())return walk(p);
 return [{path:p,kind:s.isFile()?'file':'other'}];
});}
const manifestBytes=fs.readFileSync(path.join(packet,'manifest.json'));
const manifest=JSON.parse(manifestBytes);
const checkpoint=fs.readFileSync(path.join(packet,'CHECKPOINT_SHA256SUMS'));
const readme=fs.readFileSync(path.join(packet,'README.md'));
const tree=walk(packet);
check('packet-files-only-no-symlinks',tree.every(e=>e.kind==='file'));
check('packet-ancestor-containment',noSymlinkUnder(root,path.relative(root,packet)));
const checksumEntries=checkpoint.toString().trimEnd().split('\n').map(line=>{const m=/^([a-f0-9]{64})  (.+)$/.exec(line);return m?{sha256:m[1],path:m[2]}:null;});
check('checksum-lines-valid-unique-contained',checksumEntries.every(e=>e&&safeRelative(e.path))&&new Set(checksumEntries.filter(Boolean).map(e=>e.path)).size===checksumEntries.length);
const recordedPaths=new Set(checksumEntries.filter(Boolean).map(e=>e.path));
check('checksum-inventory-exact',tree.length===checksumEntries.length+1&&tree.every(e=>path.relative(packet,e.path)==='CHECKPOINT_SHA256SUMS'||recordedPaths.has(path.relative(packet,e.path))));
check('all-checkpoint-hashes-match',checksumEntries.every(e=>e&&safeRelative(e.path)&&noSymlinkUnder(packet,e.path)&&sha(fs.readFileSync(path.join(packet,e.path)))===e.sha256));
const payloads=[];
for(const e of manifest.files){
 const valid=safeRelative(e.path)&&safeRelative(e.source)&&e.path===`evidence/${e.source}${e.encoding==='gzip'?'.gz':''}`&&['gzip','identity'].includes(e.encoding);
 if(!valid){payloads.push({source:e.source,path:e.path,valid:false});continue;}
 const stored=fs.readFileSync(path.join(packet,e.path));
 const decoded=e.encoding==='gzip'?zlib.gunzipSync(stored):stored;
 const original=fs.readFileSync(path.join(root,e.source));
 payloads.push({source:e.source,path:e.path,valid:true,encoding:e.encoding,storedHashMatches:sha(stored)===e.sha256,storedBytesMatch:stored.length===e.bytes,sourceHashMatches:sha(decoded)===e.sourceSha256,sourceBytesMatch:decoded.length===e.sourceBytes,currentSourceHashMatches:sha(original)===e.sourceSha256,exactRoundTrip:decoded.equals(original),canonicalGzip:e.encoding!=='gzip'||zlib.gzipSync(decoded,{level:9}).equals(stored),contained:noSymlinkUnder(packet,e.path)&&noSymlinkUnder(root,e.source),privateNamed:/\.private\.|(?:^|\/)credentials(?:\.|\/)|\.tar$/i.test(e.source),decoded});
}
check('manifest-paths-valid-unique-contained',payloads.every(p=>p.valid&&p.contained)&&new Set(payloads.map(p=>p.path)).size===payloads.length&&new Set(payloads.map(p=>p.source)).size===payloads.length);
check('all-stored-source-hashes-and-lengths-match',payloads.every(p=>p.valid&&p.storedHashMatches&&p.storedBytesMatch&&p.sourceHashMatches&&p.sourceBytesMatch&&p.currentSourceHashMatches));
check('every-payload-exact-original-byte-parity',payloads.every(p=>p.exactRoundTrip));
check('gzip-roundtrips-exact',payloads.filter(p=>p.encoding==='gzip').every(p=>p.exactRoundTrip));
check('gzip-streams-match-canonical-reencoding',payloads.filter(p=>p.encoding==='gzip').every(p=>p.canonicalGzip));
check('no-private-named-payloads-or-archives',payloads.every(p=>!p.privateNamed));
check('payload-inventory-exact',tree.length===payloads.length+3&&manifest.files.every(e=>recordedPaths.has(e.path))&&[...recordedPaths].every(p=>p==='README.md'||p==='manifest.json'||manifest.files.some(e=>e.path===p)));
const bySource=new Map(payloads.map(p=>[p.source,p]));
const auditSpecs=[['native-ingest-review','audit.json','reviewedFiles'],['native-ingest-review','supplement-audit.json','reviewedFiles'],['native-auth237-review','audit.json','inputs'],['native-config231-review','audit.json','inputs'],['native-config231-review','supplemental-fresh-audit.json','inputs']];
const mappings=[];
for(const [dir,file,key]of auditSpecs){
 const source=`.tmp/uat-repairs-231-246/${dir}/${file}`;
 const retained=bySource.get(source);
 check(`original-review-audit-retained:${dir}/${file}`,Boolean(retained));
 if(!retained)continue;
 const audit=JSON.parse(retained.decoded);
 for(const e of audit[key]){
  const rel=relative(e.path),payload=bySource.get(rel);
  const omissions=manifest.omitted.filter(o=>relative(o.path)===rel&&o.sha256===e.sha256);
  const copied=payload&&sha(payload.decoded)===e.sha256;
  mappings.push({audit:source,input:rel,observedSha256:e.sha256,copied:Boolean(copied),omitted:omissions.length>0,covered:Boolean(copied||omissions.length)});
 }
}
check('all-reviewed-inputs-mapped-to-exact-payload-or-explicit-omission',mappings.every(m=>m.covered));
const omissionChecks=manifest.omitted.map(o=>{
 const rel=relative(o.path),task=rel==='Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md'||rel.startsWith('backlog/tasks/task-13260.');
 const privateRecord=/\.private\.|(?:^|\/)credentials(?:\.|\/)|\.tar$/i.test(rel);
 const referenced=mappings.some(m=>m.input===rel&&m.observedSha256===o.sha256);
 const truthful=task?(/tracker|task|metadata/i.test(o.reason)&&typeof o.currentObservedSha256==='string'&&o.currentObservedSha256.length===64):privateRecord&&/private|archive/i.test(o.reason);
 const currentRecheckSha256=task?sha(fs.readFileSync(path.join(root,rel))):null;
 return {path:rel,observedSha256:o.sha256,currentObservedSha256:o.currentObservedSha256??null,currentRecheckSha256,currentMetadataHashMatches:!task||currentRecheckSha256===o.currentObservedSha256,reason:o.reason,referenced,truthful,classification:task?'tracker-task-metadata':privateRecord?'private-or-archive':'unexpected'};
});
check('omissions-are-referenced-and-truthfully-classified',omissionChecks.every(o=>o.referenced&&o.truthful));
check('omitted-metadata-current-hashes-match-at-review',omissionChecks.every(o=>o.currentMetadataHashMatches));
check('omitted-observed-version-not-mislabeled-as-copy',omissionChecks.every(o=>!bySource.has(o.path)||sha(bySource.get(o.path).decoded)!==o.observedSha256));
const directoryCoverage=[];
for(const dir of new Set(auditSpecs.map(x=>x[0]))){
 for(const name of fs.readdirSync(path.join(root,'.tmp/uat-repairs-231-246',dir)).filter(n=>/\.(md|mjs|json)$/.test(n))){
  const source=`.tmp/uat-repairs-231-246/${dir}/${name}`;
  directoryCoverage.push({source,retained:bySource.has(source)});
 }
}
check('all-original-review-packet-artifacts-retained',directoryCoverage.every(x=>x.retained));
// Known-secret values remain solely in memory. Never serialize values, variants, or credential files.
const matrix=path.join(root,'.tmp/uat-next-matrix-20260916');
const secrets=[];let credentialProfiles=0;
for(const name of fs.readdirSync(matrix).filter(n=>n.endsWith('.profile.private.json'))){
 const p=JSON.parse(fs.readFileSync(path.join(matrix,name))),c=JSON.parse(fs.readFileSync(p.credentialsPath));
 credentialProfiles++;
 secrets.push(c.apiKey,c.jwtSecret,c.apiHashSecret,...Object.values(c.accounts||{}).map(a=>a.password),...(c.providerSecrets||[]));
 if(p.pgConfigPath)secrets.push(JSON.parse(fs.readFileSync(p.pgConfigPath)).password);
}
secrets.push(JSON.parse(fs.readFileSync(path.join(root,'.tmp/fresh-uat-recovery-20260916/postgres-private.json'))).password);
const variants=[...new Set(secrets.filter(v=>typeof v==='string'&&v.length).flatMap(v=>[v,encodeURIComponent(v),JSON.stringify(v).slice(1,-1),Buffer.from(v).toString('base64'),Buffer.from(v).toString('base64url')]))];
let knownSecretMatches=0,jwtMatches=0;
const storedGzipBytes=payloads.filter(p=>p.encoding==='gzip').map(p=>fs.readFileSync(path.join(packet,p.path)));
for(const bytes of [manifestBytes,checkpoint,readme,...payloads.filter(p=>p.valid).map(p=>p.decoded),...storedGzipBytes]){
 const text=bytes.toString('utf8');
 if(variants.some(v=>text.includes(v)))knownSecretMatches++;
 if(/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/.test(text))jwtMatches++;
}
check('independent-known-secret-and-jwt-scan-clean',knownSecretMatches===0&&jwtMatches===0);
check('expected-accepted-findings-only',JSON.stringify(manifest.acceptedFindings)===JSON.stringify([231,233,237,244,245,247])&&manifest.fullMatrixAccepted===false);
const summary=readme.toString('utf8');
check('material-limits-present',summary.includes('analysis was truncated and did not succeed')&&summary.includes('No native hard-quota/concurrency')&&summary.includes('UAT238')&&summary.includes('257')&&summary.includes('258')&&summary.includes('neither releases nor accepts')&&summary.includes('missing-key-only231 capture was insufficient')&&summary.includes('Settled UI snapshots do not certify every transient event or zero requests'));
const reportPath=path.join(here,'REVIEW.md');
const result={observedAt:new Date().toISOString(),verdict:checks.every(c=>c.passed)?'CLEAR_BOUNDED_RETENTION':'REQUIRES_CORRECTION',packet:path.relative(root,packet),packetManifestSha256:sha(manifestBytes),checkpointSha256:sha(checkpoint),readmeSha256:sha(readme),initialReviewedManifestSha256:'fe005dde4513b7c64521a68ac669e14ac47ece9c7f4a8ed6d7dca9fb4941ce74',reviewCorrection:'Public task179 filename matched broad credentials filter; controller corrected metadata classification and retained observed/current hashes before final verification.',counts:{payloads:payloads.length,totalFiles:tree.length,gzip:payloads.filter(p=>p.encoding==='gzip').length,checksumEntries:checksumEntries.length,reviewedInputReferences:mappings.length,omissions:manifest.omitted.length},checks,allChecksPass:checks.every(c=>c.passed),payloads:payloads.map(({decoded,...entry})=>entry),reviewInputMappings:mappings,omissionChecks,originalReviewArtifactCoverage:directoryCoverage,secretScan:{credentialProfiles,knownVariants:variants.length,decodedAndMetadataFilesScanned:payloads.length+3,storedGzipFilesAlsoScanned:storedGzipBytes.length,knownSecretMatches,jwtMatches},reviewFiles:{scriptSha256:sha(fs.readFileSync(fileURLToPath(import.meta.url))),reportSha256:fs.existsSync(reportPath)?sha(fs.readFileSync(reportPath)):null},retainerSha256:sha(fs.readFileSync(path.join(root,'.tmp/uat-repairs-231-246/retain-config-ingest-auth-native.mjs'))),limitations:['Known-secret and JWT scans are scoped, not a proof that every possible secret pattern is absent','Private/archive omissions retain observed hashes without original-byte parity','Mutable tracker/task hashes are historical observations','Nested repository source manifests retained exactly; unretained repository trees were not exhaustively rehashed','No product/runtime/native-acceptance reevaluation beyond retained original verdicts']};
fs.writeFileSync(path.join(here,'audit.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({verdict:result.verdict,manifestSha256:result.packetManifestSha256,counts:result.counts,checks:checks.length,passed:checks.filter(c=>c.passed).length,failed:checks.filter(c=>!c.passed).map(c=>c.name),knownSecretMatches,jwtMatches,auditSha256:sha(fs.readFileSync(path.join(here,'audit.json')))}));
