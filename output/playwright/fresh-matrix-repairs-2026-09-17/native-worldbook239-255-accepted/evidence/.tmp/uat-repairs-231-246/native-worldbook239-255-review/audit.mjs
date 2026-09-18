import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const root=process.cwd(),out='.tmp/uat-repairs-231-246/native-worldbook239-255-review';
const base='.tmp/uat-repairs-231-246/native-targeted';
const revision='2787043410fc918b2c280d90f753d8fd02b6b35b';
const manager='tldw_Server_API/app/core/Character_Chat/world_book_manager.py';
const expectedManager='fbefcf2e2283b8a8ed26de6c0829e7104c53d187560713b0e833ac5e0bf90d5f';
const inputs=[],checks=[];const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function bytes(p,privateHashOnly=false){p=path.resolve(root,p);const b=fs.readFileSync(p);inputs.push({path:path.relative(root,p),bytes:b.length,sha256:sha(b),privateHashOnly});return b;}
const json=(p,priv=false)=>JSON.parse(bytes(p,priv));
const check=(name,pass)=>checks.push({name,pass:Boolean(pass)});
function capture(name,cell='pg-multi'){const text=bytes(`${base}/${cell}/${name}`).toString();const m=text.match(/### Result\n([\s\S]*?)(?:\n### |$)/);let value=null;try{value=JSON.parse(m?.[1]||'null')}catch{};return {text,value};}
const nativeNames=['worldbook255-open.txt','worldbook255-catalog-settled.txt','worldbook255-create-open.txt','worldbook255-created.txt','worldbook255-created-observed.txt','worldbook255-edit.txt','worldbook255-reload.txt','worldbook255-upgrade-reload.txt','worldbook239-created.txt','worldbook239-characters-entry.txt','worldbook239-characters-settled.txt','worldbook239-editor-open.txt','worldbook239-editor-observed.txt','worldbook239-editor-closed.txt'];
const c=Object.fromEntries(nativeNames.map(n=>[n,capture(n)]));
const ev=n=>c[n].value?.events||[];
const url=e=>{try{return new URL(e.url).pathname+new URL(e.url).search}catch{return ''}};
const wb='/api/v1/characters/world-books';
const created=ev('worldbook255-created-observed.txt').find(e=>e.status===201&&url(e)===wb);
const createRequest=ev('worldbook255-created-observed.txt').find(e=>e.method==='POST'&&url(e)===wb);
const listed=ev('worldbook255-created-observed.txt').find(e=>e.status===200&&url(e)===wb);
const reloaded=ev('worldbook255-reload.txt').find(e=>e.status===200&&url(e)===wb);
const editor=ev('worldbook239-editor-observed.txt').find(e=>e.status===200&&url(e)===wb+'?include_disabled=true');
const associations=ev('worldbook239-editor-observed.txt').find(e=>e.status===200&&url(e)==='/api/v1/characters/4/world-books');
const record=created?.body;
const expected={name:'UAT239 Alice public fictional context',description:'Synthetic public catalog acceptance fixture. Rowan Observatory is fictional.',scan_depth:3,token_budget:500,recursive_scanning:false,enabled:true};
const exactFields=row=>row&&Object.entries(expected).every(([k,v])=>row[k]===v);
check('Native successful POST201 matches submitted public fixture',created?.at==='2026-09-17T23:51:02.399Z'&&createRequest?.at==='2026-09-17T23:51:02.360Z'&&exactFields(createRequest.body)&&exactFields(record)&&record.id===1&&record.entry_count===0);
check('Immediate catalog GET200 retains exactly the created record',listed?.body?.total===1&&JSON.stringify(listed.body.world_books?.[0])===JSON.stringify(record));
check('Normal reload GET200 retains exactly the created record',reloaded?.at==='2026-09-17T23:54:20.086Z'&&reloaded?.body?.total===1&&JSON.stringify(reloaded.body.world_books?.[0])===JSON.stringify(record));
const editUI=c['worldbook255-edit.txt'].value?.ui||'';
check('Edit UI reads back name description and enabled state',editUI.includes(expected.name)&&editUI.includes(expected.description)&&/switch "Enabled Enabled" \[checked\]/.test(editUI));
check('Character 4 dependent catalog returns the same populated book',editor?.at==='2026-09-17T23:55:56.631Z'&&editor.body.total===1&&JSON.stringify(editor.body.world_books?.[0])===JSON.stringify(record));
check('Character association GET200 returns empty list without edit mutation',associations?.at==='2026-09-17T23:55:56.714Z'&&Array.isArray(associations.body)&&associations.body.length===0&&!ev('worldbook239-editor-observed.txt').some(e=>e.at>='2026-09-17T23:54:53'&&e.method&&!['GET','HEAD','OPTIONS'].includes(e.method)));
check('Character editor is visibly closed',!/dialog "Edit character"/.test(c['worldbook239-editor-closed.txt'].value?.ui||'')&&c['worldbook239-editor-closed.txt'].value?.url.endsWith('/characters'));
check('Earlier create500 remains retained as failed attempt',ev('worldbook239-created.txt').some(e=>e.status===500&&e.at==='2026-09-17T22:37:24.025Z'&&url(e)===wb));
check('Separate UAT260 future-time display is retained',/in 7 hours/.test(c['worldbook255-reload.txt'].value?.ui||''));
const helpers=['worldbook255-create.js','worldbook255-edit.js','worldbook255-open.js','worldbook255-reload.js','worldbook239-characters-entry.js','worldbook239-close-editor.js'].map(n=>bytes(`${base}/pg-multi/${n}`).toString());
check('Reviewed helpers use normal UI/reload and no response interception',helpers.some(s=>s.includes('await page.reload()'))&&helpers.every(s=>!/(?:route\(|fulfill\(|setExtraHTTPHeaders|addCookies|localStorage|sessionStorage|request\.(?:post|put|delete))/.test(s)));
const empty=[];for(const cell of ['pg-single','pg-multi']){const r=capture('testbot-created-entry.txt',cell);const response=r.value?.events?.find(e=>e.status===200&&url(e)===wb+'?include_disabled=true');check(`${cell} earlier empty editor catalog returned200`,response?.body?.total===0&&response?.body?.world_books?.length===0);empty.push({cell,at:response?.at,status:response?.status,count:response?.body?.total});}
const startup=json('.tmp/uat-repairs-231-246/native-upgrade-preparation/retry256-startup-safe.json');
const matrix=[];
for(const cell of ['pg-single','pg-multi']){
 const processes=startup.processes.filter(p=>p.cell===cell),sourceRoot=processes[0].sourceRoot;
 const bindingPath=path.join(root,'.tmp/uat-next-matrix-20260916/targeted-upgrades/retry256-worldbook255-upgrade-20260917',cell,'binding.private.json');
 const bb=bytes(bindingPath,true),binding=JSON.parse(bb);const pb=bytes(binding.originalProfile,true),profile=JSON.parse(pb);
 const ib=bytes(path.join(profile.root,'initialized.private.json'),true),initialization=JSON.parse(ib);
 const hb=bytes(profile.pgReceiptPath,true),holder=JSON.parse(hb);
 const mb=bytes(path.join(path.dirname(sourceRoot),'preparation',`${cell}-source-manifest.json`)),manifest=JSON.parse(mb);
 const sb=bytes(path.join(sourceRoot,manager));
 const receiptProof=processes.map(p=>{const rb=bytes(p.receipt,true),r=JSON.parse(rb);return {action:p.action,pid:p.pid,start:p.startedAt,currentStatus:r.status,observedReceiptSha256:sha(rb),initialReceiptSha256:p.sha256,immutableFieldsMatch:r.bindingHash===sha(bb)&&r.pid===p.pid&&r.sourceRoot===sourceRoot&&r.sourceCommit===revision&&r.startedAt===p.startedAt,coversAcceptance:r.startedAt<'2026-09-17T23:51:02.360Z'&&(!r.endedAt||r.endedAt>'2026-09-17T23:56:51.898Z')};});
 check(`${cell} source/binding/manifest manager parity`,binding.sourceCommit===revision&&manifest.revision===revision&&binding.sourceRoot===sourceRoot&&binding.proof.sourceManifest===sha(mb)&&manifest.files.find(e=>e.path===manager)?.sha256===sha(sb)&&sha(sb)===expectedManager);
 check(`${cell} original profile initialization holder preserved`,sha(pb)===binding.originalProfileHash&&sha(ib)===binding.originalInitializationHash&&sha(hb)===binding.originalHolderHash&&initialization.status==='completed'&&initialization.preparationHash===sha(JSON.stringify(profile))&&holder.status==='held'&&holder.source_root===profile.sourceRoot&&holder.source_commit===profile.sourceCommit);
 check(`${cell} API/frontend receipts cover native acceptance`,receiptProof.length===2&&receiptProof.every(r=>r.immutableFieldsMatch&&r.coversAcceptance));
 matrix.push({cell,sourceSha256:sha(sb),bindingSha256:sha(bb),receiptProof});
 if(cell==='pg-multi'){
  const credentials=json(profile.credentialsPath,true),alice=credentials.accounts.alice;
  const before=capture('upgrade256-readback.txt').value;
  const beforeIdentity=before.events.filter(e=>e.event==='identity'&&e.status===200).at(-1);
  const reloadIdentity=ev('worldbook255-reload.txt').filter(e=>e.event==='identity'&&e.status===200).at(-1);
  const editorIdentity=ev('worldbook239-editor-observed.txt').filter(e=>e.event==='identity'&&e.status===200).at(-1);
  const identities=[beforeIdentity,reloadIdentity,editorIdentity];
  check('Post-upgrade and reload/editor authenticated identities match private Alice2 fixture',identities.every(e=>e?.identity?.id===2&&e.identity.username===alice.username));
  check('Alice identity observations surround create and precede character catalog',beforeIdentity?.at<'2026-09-17T23:51:02.360Z'&&reloadIdentity?.at>'2026-09-17T23:51:02.399Z'&&reloadIdentity?.at<'2026-09-17T23:54:20.086Z'&&editorIdentity?.at<'2026-09-17T23:55:56.631Z');
  matrix.at(-1).identity={userId:2,observedAt:identities.map(e=>e.at),usernameMatchesPrivateFixture:identities.every(e=>e.identity.username===alice.username)};
 }
}
const committed=[];for(const ref of ['15c1','2787043410']){const rev=execFileSync('git',['rev-parse',ref+'^{commit}'],{encoding:'utf8'}).trim();const b=execFileSync('git',['show',rev+':'+manager]);committed.push({revision:rev,sha256:sha(b)});}
check('Manager bytes equal committed15c1 and278 revisions',committed.every(c=>c.sha256===expectedManager)&&sha(bytes(manager))===expectedManager);
for(const p of ['.tmp/uat-repairs-231-246/worldbook255-review/round1-REVIEW.md','.tmp/uat-repairs-231-246/worldbook255-review/round1-audit.json','.tmp/uat-repairs-231-246/review234-239/REVIEW.md'])bytes(p);
bytes(`${out}/audit.mjs`);
const audit={task:'UAT239 / TASK13260.181 and UAT255 / TASK13260.197',at:new Date().toISOString(),verdict:checks.every(c=>c.pass)?'CLEAR bounded native catalog/create/readback':'GAPS',checks,summary:{emptyCatalogs:empty,created:{at:created.at,id:record.id,version:record.version,enabled:record.enabled,scanDepth:record.scan_depth,tokenBudget:record.token_budget,recursiveScanning:record.recursive_scanning,entryCount:record.entry_count,recordSha256:sha(JSON.stringify(record)),nameSha256:sha(record.name),descriptionSha256:sha(record.description)},reloadAt:reloaded.at,editorAt:editor.at,associationCount:associations.body.length,timestampIssue260Open:true,matrix,committed},limits:['Fresh populated create/reload/editor is PG-multi Alice only; earlier empty catalogs cover both PG modes under prior source.','Authenticated identity observer events before create and on reload/editor match private Alice2; the create response itself has no separate identity field.','No entries were created, no character association was saved, and no duplicate/conflict/ownership/CRUD certification is claimed.','Timestamp display remains incorrect (UAT260); chronology is not accepted.','Earlier create500 and startup launch/navigation/transient500 failures remain retained.','Process receipts may change status on exit; immutable bindings and observed receipt hashes are reported separately.','No tests or native actions were rerun; only source/receipt/evidence reads and this review packet were performed. Read-only Git object lookups verify committed source bytes.','Private records are read only in memory and hash-only in inputs; unrelated events, provider reasoning, full UI and credentials are not serialized.'],inputs};
fs.mkdirSync(out,{recursive:true});fs.writeFileSync(`${out}/audit.json`,JSON.stringify(audit,null,2)+'\n');console.log(JSON.stringify({verdict:audit.verdict,checks:checks.length,failed:checks.filter(c=>!c.pass),inputs:inputs.length,sha256:sha(fs.readFileSync(`${out}/audit.json`))}));
