import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root = process.cwd();
const base = '.tmp/uat-repairs-231-246/native-targeted';
const out = '.tmp/uat-repairs-231-246/native-ingest-review';
const files = new Set();
const checks = [];
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
function read(p) { files.add(p); return fs.readFileSync(path.join(root,p),'utf8'); }
function result(rel) { const s=read(`${base}/${rel}`); const m=s.match(/### Result\n([\s\S]*?)(?:\n### |$)/); if(!m) throw Error(`No result: ${rel}`);return JSON.parse(m[1]); }
function check(name,pass) {checks.push({name,pass:Boolean(pass)});}
const cases = [
 {name:'single', prefix:'pg-single/ingest', min:'pg-single/ingest-minimize-media.txt', owner:1,job:1,media:1,uuid:'d4a0d5e2-6e9a-4d7a-bc9e-f011af4149d0',fixture:'rowan-observatory-public-20260917.txt',detail:'pg-single/media-upgraded-read-v2.txt'},
 {name:'alice',prefix:'pg-multi/alice-ingest',min:'pg-multi/alice-ingest-minimized-media.txt',owner:2,job:4,media:1,uuid:'be7b0cbc-a4db-4036-8d53-e1c6f481f79d',fixture:'rowan-observatory-public-20260917.txt',detail:'pg-multi/alice-media-selected.txt'},
 {name:'bob',prefix:'pg-multi/bob-ingest',min:'pg-multi/bob-ingest-minimized-media.txt',owner:3,job:5,media:2,uuid:'bfbd4ff9-2f2e-4e6d-bc7a-a9e0886ae6d4',fixture:'birch-workshop-bob-public-20260917.txt',detail:'pg-multi/bob-own-media-selected.txt'}
];
const jobs=cases.map(c=>{
 const started=result(`${c.prefix}-start.txt`),min=result(c.min),d=result(`${c.prefix}-observed.txt`);
 const end=d.events.filter(e=>e.event==='response'&&e.status===200&&e.body?.id===c.job&&e.body?.status==='completed');
 const e=end.at(-1),j=e?.body,r=j?.result;
 const admissions=d.events.filter(e=>e.status===200&&e.url?.endsWith('/media/ingest/jobs')&&e.body?.jobs?.some(j=>j.id===c.job));
 const details=result(c.detail);const detailEvent=details.events.find(e=>e.status===200&&e.url?.endsWith(`/media/${c.media}`));const detail=detailEvent?.body;
 const fixture=read(`.tmp/uat-next-matrix-20260916/fixtures/${c.fixture}`);const content=detail?.content?.text;
 check(`${c.name}: real queued admission`,admissions.length===1&&admissions[0].body.errors.length===0);
 check(`${c.name}: completed correct job owner/media/UUID`,j?.status==='completed'&&+j.owner_user_id===c.owner&&r?.media_id===c.media&&r?.media_uuid===c.uuid&&j.progress_percent===100);
 check(`${c.name}: exactly one terminal truncation warning`,end.length>0&&end.every(e=>e.body.result.status==='Warning'&&e.body.result.error===null&&e.body.result.warnings.length===1&&e.body.result.warnings[0].includes('Provider analysis was truncated')));
 check(`${c.name}: minimized before completion`,Date.parse(min.at)<Date.parse(`${j?.completed_at.replace(' ','T')}Z`));
 const minText=read(`${base}/${c.min}`);
 check(`${c.name}: actual minimize and Media navigation receipt`,minText.includes("name:'Minimize to Background'")&&minText.includes("name:'Media'"));
 const observedText=read(`${base}/${c.prefix}-observed.txt`);const observationCode=observedText.match(/### Ran Playwright code\n([\s\S]*?)(?:\n### |$)/)?.[1]||'';
 check(`${c.name}: observation only, no reload/search/click`,observationCode.includes('ariaSnapshot')&&!/\.click\(|\.reload\(|\.goto\(|\.fill\(/.test(observationCode));
 check(`${c.name}: owned catalogue one result`,d.body.includes('Results 1 / 1')&&d.body.includes(`Select media: ${c.fixture.replace('.txt','')}`));
 check(`${c.name}: own detail exact complete fixture minus final newline`,detail?.media_id===c.media&&content===fixture.replace(/\n$/,''));
 check(`${c.name}: source saved and chunked without analysis success`,detail?.processing?.analysis===null&&detail.processing.chunking_status==='completed');
 return {name:c.name,jobId:j.id,ownerUserId:+j.owner_user_id,mediaId:r.media_id,mediaUuid:r.media_uuid,admittedAt:admissions[0].at,minimizedAt:min.at,completedAt:j.completed_at,firstTerminalResponseAt:end[0].at,terminalReadCount:end.length,observedAt:d.at,resultStatus:r.status,warnings:r.warnings,detailAt:detailEvent.at,contentLength:content.length,contentSha256:sha(content),chunkingStatus:detail.processing.chunking_status,analysisPresent:detail.processing.analysis!==null,sourceVersionCreated:detail.versions?.[0]?.created_at,emptyBefore:min.body.includes('Results 0 / 0'),listingRequestsRetained:d.events.filter(e=>/\/api\/v1\/media\/?(?:\?|$)/.test(e.url)).length};
});
check('multi: two owners persisted distinct media identities',jobs[1].mediaId!==jobs[2].mediaId&&jobs[1].mediaUuid!==jobs[2].mediaUuid&&jobs[1].ownerUserId!==jobs[2].ownerUserId);
const bobForeign=result('pg-multi/bob-foreign-alice-media-captured.txt');
const aliceForeign=result('pg-multi/alice-foreign-bob-media-v2.txt');
const login=result('pg-multi/alice-ownership-login.txt');
read(`${base}/pg-multi/bob-ownership-logout.txt`);
const identities=login.auth.filter(e=>e.event==='identity').map(e=>({at:e.at,status:e.status,id:e.identity.id,username:e.identity.username}));
check('Bob identity 3 corroborates foreign and own reads',identities.some(i=>i.id===3&&i.at.startsWith('2026-09-17T22:14:51'))&&identities.some(i=>i.id===3&&i.at.startsWith('2026-09-17T22:25:34')));
check('Alice identity 2 after ordinary re-login',identities.at(-1)?.id===2&&identities.at(-1)?.status===200);
check('Bob cannot read Alice Media 1',bobForeign.events.some(e=>e.url?.endsWith('/media/1')&&e.status===404)&&bobForeign.ui.includes('Results 0 / 0'));
check('Alice cannot read Bob Media 2',aliceForeign.status===404&&aliceForeign.events.some(e=>e.url?.endsWith('/media/2')&&e.status===404));
check('Alice reciprocal own catalogue has only Rowan',aliceForeign.ui.includes('Results 1 / 1')&&aliceForeign.ui.includes('Select media: rowan-observatory-public-20260917')&&!aliceForeign.ui.includes('birch-workshop-bob-public-20260917'));
check('Bob own catalogue excludes Alice source',!result('pg-multi/bob-ingest-observed.txt').body.includes('rowan-observatory-public-20260917'));
const excluded=[];
for(const rel of ['pg-single/ingest-open-settled.txt','pg-multi/alice-ingest-open.txt','pg-multi/alice-foreign-bob-media.txt']){
 const s=read(`${base}/${rel}`);excluded.push({file:`${base}/${rel}`,pageUrl:s.match(/- Page URL: (.+)/)?.[1]??null,referenceError:/ReferenceError/.test(s),reason:rel.includes('foreign')?'Initial helper failure excluded; use corrected v2.':'Landed in Chat; not an Open-in-Media success.'});
}
for(const p of ['Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md',`${base}/RUN.md`,`${base}/UPGRADED_NATIVE_NOTES.md`,'.tmp/uat-repairs-231-246/native-preparation/ACCEPTANCE.md','.tmp/uat-repairs-231-246/review233-238-247/REVIEW.md','.tmp/uat-repairs-231-246/review245/REVIEW.md','.tmp/uat-repairs-231-246/review-media/REVIEW.md','.tmp/uat-repairs-231-246/native-media-review/REVIEW.md','.tmp/uat-repairs-231-246/native-media-review/audit.json','.tmp/uat-repairs-231-246/upgrade254-native-review/REVIEW.md','.tmp/uat-repairs-231-246/upgrade254-native-review/audit.json'])read(p);
for(const p of fs.readdirSync('backlog/tasks').filter(p=>/^task-13260\.(175|180|186|187|189) /.test(p)))read(`backlog/tasks/${p}`);
const audit={at:new Date().toISOString(),scope:['UAT233','UAT238','UAT244','UAT245','UAT247'],run:'repairs231-250-targeted-20260917',upgradedSource:'a7d3155a567afb25982eb360ea24b973cc3249c9',sourceBinding:'Reused independently reviewed upgrade254 actual startup/preservation report; no fresh process or runtime inspection.',checks,passed:checks.filter(x=>x.pass).length,failed:checks.filter(x=>!x.pass),jobs,identityTimeline:identities,foreignReadControls:{bob:{file:`${base}/pg-multi/bob-foreign-alice-media-captured.txt`,at:bobForeign.at,status:404},alice:{file:`${base}/pg-multi/alice-foreign-bob-media-v2.txt`,at:aliceForeign.at,status:aliceForeign.status}},excluded,limitations:['Native Quick Ingest terminal Results panel warning label was not recaptured; terminal Warning responses and saved source are proven.','Minimize→Media automatic refresh is supported; literal wizard Open in Media action was not successfully captured.','Observer filtered out bare /media list URLs, so exact list-fetch time is not established.','Single/Alice immediate post-click body snapshots still show prior Chat; receipt records Media navigation before completion. Bob has an actual empty catalogue baseline.','No native rerun of deliberate search/filter preservation, hard-quota rejection, concurrent sequence allocation, SQLite cells or full matrix; use separately reviewed controlled coverage for those branches.','UAT238 source-dependent full-source Chat succeeds in the separate native-media audit; all QA/reanalysis/Trash workflows are not certified here.'],reviewedFiles:[...files].sort().map(p=>({path:p,bytes:fs.statSync(p).size,sha256:sha(fs.readFileSync(p))}))};
fs.writeFileSync(path.join(root,out,'audit.json'),JSON.stringify(audit,null,2)+'\n');
console.log(JSON.stringify({passed:audit.passed,failed:audit.failed,hashedFiles:audit.reviewedFiles.length,auditSha256:sha(fs.readFileSync(path.join(root,out,'audit.json')))}));
if(audit.failed.length)process.exitCode=1;
