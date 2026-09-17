import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const out='.tmp/uat-repairs-231-246/native-ingest-review';
const base='.tmp/uat-repairs-231-246/native-targeted/pg-multi';
const inputs=new Set(),checks=[];
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const hash=p=>{inputs.add(p);return sha(fs.readFileSync(p));};
const read=p=>{inputs.add(p);return fs.readFileSync(p,'utf8');};
const json=p=>JSON.parse(read(p));
function result(name){const s=read(`${base}/${name}`);return JSON.parse(s.match(/### Result\n([\s\S]*?)(?:\n### |$)/)[1]);}
function check(name,pass){checks.push({name,pass:Boolean(pass)});}
const oldExpected={
 'REVIEW.md':'5087e4c38f749f0fb9302d6dcdb772f0936b21b0e5b77e292123131e902867f5',
 'audit.mjs':'dff52efc261fd7c661fdee5fde0abecdb2b9aaf4ebc4fbdc2bb9933f3a2739a1',
 'audit.json':'4dd68f60a13902f62b5c9b87e7d0f8c3549d5f3d7e8c426bc599f1724aa8aab6'
};
for(const [name,expected]of Object.entries(oldExpected))check(`Original ${name} unchanged`,hash(`${out}/${name}`)===expected);
const terminal=result('results-ingest-terminal.txt'),opened=result('results-open-media.txt'),settled=result('results-open-media-settled.txt');
const started=result('results-ingest-started.txt'),observed=result('results-ingest-observed.txt');
for(const f of ['results-ingest-resume-snapshot.txt','results-ingest-provider-options.txt','results-ingest-review.txt'])read(`${base}/${f}`);
const end=terminal.events.filter(e=>e.status===200&&e.body?.id===6&&e.body.status==='completed');
const j=end.at(-1)?.body,r=j?.result,w=r?.warnings?.[0];
check('Job6 owned by Alice2, saved Media3 expected UUID',j?.owner_user_id==='2'&&r?.media_id===3&&r?.media_uuid==='3936bff9-39ab-4fcd-8e18-d07a2900259b');
check('Two terminal readbacks keep one genuine truncation warning',end.length===2&&end.every(e=>e.body.result.status==='Warning'&&e.body.result.warnings.length===1&&e.body.result.warnings[0]===w&&e.body.result.error===null)&&w?.includes('Provider analysis was truncated before completion'));
const admission=terminal.events.find(e=>e.status===200&&e.url?.endsWith('/media/ingest/jobs')&&e.body.jobs?.some(j=>j.id===6));
check('Actual enqueue HTTP200, no admission errors',admission&&admission.body.errors.length===0);
check('Processing observation precedes completion',Date.parse(observed.at)<Date.parse(j.completed_at.replace(' ','T')+'Z'));
check('Native Results region explicitly saved-with-warnings1',terminal.body.includes('region "Items saved with warnings"')&&terminal.body.includes('heading "Saved with warnings (1)"'));
check('Warning paragraph appears exactly once',terminal.body.split(w).length-1===1);
check('No clean-success/failure/cancel claim',terminal.body.includes('Total: 0 succeeded, 1 saved with warnings, 0 skipped, 0 not submitted, 0 failed, 0 cancelled'));
check('Catalogue already contains two owned sources before Open',terminal.body.includes('Results 2 / 2')&&terminal.body.includes('Select media: linden-results-public-20260917')&&terminal.body.includes('Select media: rowan-observatory-public-20260917')&&!terminal.body.includes('birch-workshop-bob-public-20260917'));
const helper=read(`${base}/results-open-media.js`);
const receipt=read(`${base}/results-open-media.txt`).match(/### Ran Playwright code\n([\s\S]*?)(?:\n### |$)/)?.[1]||'';
check('Exact enabled Results button actually clicked',terminal.body.includes('button "Open linden-results-public-20260917.txt in Media": Open in Media')&&helper.includes("name:'Open linden-results-public-20260917.txt in Media',exact:true}).click()")&&receipt.includes(helper.trim()));
check('Click reaches exact Media3 with wizard hidden',opened.url==='http://127.0.0.1:18783/media?id=3'&&!opened.body.includes('dialog "Quick Ingest"')&&opened.body.includes('heading "linden-results-public-20260917"'));
const details=settled.events.filter(e=>e.status===200&&e.url?.endsWith('/media/3'));
const fixture=read('.tmp/uat-next-matrix-20260916/fixtures/linden-results-public-20260917.txt');
const source=fixture.replace(/\n$/,'');
check('Both actual detail HTTP200 responses contain exact full fixture',details.length===2&&details.every(e=>e.body.media_id===3&&e.body.content?.text===source));
check('First detail follows actual click',Date.parse(details[0]?.at)>=Date.parse(opened.startedAt));
const region=settled.body.match(/  - region "Media content":\n([\s\S]*?)\n  - button "Analysis"/)?.[1]||'';
const paragraphs=[...region.matchAll(/^    - paragraph: (.+)$/gm)].map(m=>m[1].startsWith('"')?JSON.parse(m[1]):m[1]);
check('Settled rendered content paragraphs equal complete source',paragraphs.join('\n\n')===source);
check('Settled route, selection and catalogue remain correct',read(`${base}/results-open-media-settled.txt`).includes('- Page URL: http://127.0.0.1:18783/media?id=3')&&settled.body.includes('Showing linden-results-public-20260917')&&settled.body.includes('Results 2 / 2')&&!settled.body.includes('Get started — ingest your first content'));
check('Saved warning never becomes successful analysis',settled.body.includes('No analysis yet')&&details.every(e=>e.body.processing.analysis===null&&e.body.processing.chunking_status==='completed'));
const observationCode=read(`${base}/results-open-media-settled.txt`).match(/### Ran Playwright code\n([\s\S]*?)(?:\n### |$)/)?.[1]||'';
check('Open and settled observation do not reload or search',!/[.]reload\(|[.]goto\(|[.]fill\(/.test(helper+observationCode));
const run='model232-upgrade-20260917',root='.tmp/uat-next-matrix-20260916';
const receiptDir=`${root}/targeted-upgrades/${run}/pg-multi`,prep=`${root}/repair-sources/${run}/preparation`;
const bindingPath=`${receiptDir}/binding.private.json`,binding=json(bindingPath),gate=json('.tmp/uat-repairs-231-246/native-upgrade-preparation/model232-gate.json'),completion=json(`${prep}/complete.json`);
const expectedRevision='6f6983b0620aae1f0892c6b0d3ae3bebfc105e02';
check('Released gate, binding and completion agree on upgraded source',gate.status==='RELEASED'&&[gate.revision,binding.sourceCommit,completion.revision].every(x=>x===expectedRevision)&&binding.originalRun==='repairs231-250-targeted-20260917'&&binding.upgradeRun===run);
for(const [key,file]of Object.entries({gate:'gate.json',completion:'complete.json',sourceManifest:'pg-multi-source-manifest.json',pythonReuse:'python-reuse.json'}))check(`Binding ${key} hash matches retained proof`,hash(`${prep}/${file}`)===binding.proof[key]);
const launches=fs.readdirSync(receiptDir).filter(f=>f.endsWith('.process.private.json')).map(f=>json(`${receiptDir}/${f}`));
check('Both recorded launch receipts bind exact source before ingestion',launches.length===2&&launches.every(p=>p.status==='started'&&p.sourceCommit===expectedRevision&&p.bindingHash===hash(bindingPath)&&Date.parse(p.startedAt)<Date.parse(started.at)));
const oldBindingPath=`${root}/targeted-upgrades/repairs251-254-upgrade2-20260917/pg-multi/binding.private.json`;
const priorBinding=json(oldBindingPath);
check('Original profile/init/holder fingerprints unchanged across upgrades',['originalProfileHash','originalInitializationHash','originalHolderHash'].every(k=>binding[k]===priorBinding[k]));
const audit={at:new Date().toISOString(),purpose:'Supplement only: native233 Results warning and244 literal Open in Media gaps.',checks,passed:checks.filter(c=>c.pass).length,failed:checks.filter(c=>!c.pass),oldPacketExpected:oldExpected,source:{revision:expectedRevision,upgradeRun:run,originalRun:binding.originalRun,recordedLaunches:launches.map(p=>({action:p.action,status:p.status,pid:p.pid,startedAt:p.startedAt})),limit:'Retained source/launch receipts verified; no live process inspection or full copied-source rehash.'},native:{jobId:j.id,ownerUserId:+j.owner_user_id,mediaId:r.media_id,mediaUuid:r.media_uuid,admittedAt:admission.at,completedAt:j.completed_at,terminalAt:terminal.at,terminalResponses:end.map(e=>e.at),warnings:r.warnings,clickStartedAt:opened.startedAt,clickReceiptAt:opened.at,settledAt:settled.at,route:opened.url,detailResponses:details.map(e=>({at:e.at,status:e.status})),contentLength:source.length,contentSha256:sha(source),renderedParagraphCount:paragraphs.length,analysisPresent:false},remainingLimits:['UAT233 and244 previously missing visible/button gaps are now supplied on PG multi. Existing reviewed controlled tests and prior native owner checks remain separate evidence.','Original SQLite Bob/admin and deliberate active-filter native permutations were not rerun; exact list-fetch timestamp remains unavailable.','Native concurrent allocations and quota rejection are not newly tested.','UAT238 source QA/reanalysis/sole-item Trash dependencies remain outside this supplement.'],reviewedFiles:[...inputs].sort().map(p=>({path:p,bytes:fs.statSync(p).size,sha256:sha(fs.readFileSync(p))}))};
fs.writeFileSync(`${out}/supplement-audit.json`,JSON.stringify(audit,null,2)+'\n');
console.log(JSON.stringify({passed:audit.passed,failed:audit.failed,hashedFiles:audit.reviewedFiles.length,auditSha256:sha(fs.readFileSync(`${out}/supplement-audit.json`))}));
if(audit.failed.length)process.exitCode=1;
