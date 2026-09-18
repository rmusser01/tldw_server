import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const out='.tmp/uat-repairs-231-246/capability258-native-diagnosis';
const base='.tmp/uat-repairs-231-246/native-targeted';
const run='authscope257259-cap258-upgrade-20260918';
const revision='edfd06ec40a173f2e38ec65af715abb29f3aa002';
const oldRun='.tmp/uat-next-matrix-20260916/repair-sources/retry256-worldbook255-upgrade-20260917/pg-multi/';
const inputs=[],checks=[],sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function bytes(p,privateHashOnly=false){const b=fs.readFileSync(p);inputs.push({path:p,bytes:b.length,sha256:sha(b),privateHashOnly});return b;}
const json=(p,priv=false)=>JSON.parse(bytes(p,priv));
const check=(name,pass)=>checks.push({name,pass:Boolean(pass)});
function capture(cell,n){const s=bytes(`${base}/${cell}/${n}.txt`).toString(),m=s.match(/### Result\n([\s\S]*?)(?:\n### |$)/);return JSON.parse(m?.[1]||'null');}
const endpoint=e=>{try{return new URL(e.url).pathname}catch{return ''}};
const fresh={};
for(const cell of ['pg-single','pg-multi']){
 const c=capture(cell,'capability258-fresh'),helper=bytes(`${base}/${cell}/capability258-fresh.js`).toString();
 fresh[cell]={at:c.at,observedAt:c.observedAt,events:c.events.length,capabilityRequests:c.events.filter(e=>e.event==='request'&&endpoint(e)==='/api/v1/ingestion-sources/capabilities').length,errors:c.messages.filter(m=>['error','pageerror'].includes(m.type)).length,credentialGateVisible:c.ui.includes('Add your credentials to use Media')};
 check(`${cell} fresh context comprehensive observer records no protected capability dispatch/errors`,helper.includes("r.url().includes('/api/')")&&helper.includes('browser().newContext()')&&!/addCookies|localStorage|setExtraHTTPHeaders|fulfill\(/.test(helper)&&fresh[cell].events>0&&fresh[cell].capabilityRequests===0&&fresh[cell].errors===0&&fresh[cell].credentialGateVisible);
}
const captures=Object.fromEntries(['authenticated','authenticated-reloaded','sources','sources-settled','sources-retry'].map(n=>[n,capture('pg-multi','capability258-'+n)]));
const nativeSummary=Object.fromEntries(Object.entries(captures).map(([n,c])=>[n,{at:c.at,observedAt:c.observedAt,identity:c.events.filter(e=>e.event==='identity').map(e=>({at:e.at,status:e.status,userId:e.identity?.id})),authLifetime:c.events.filter(e=>e.event==='auth-lifetime').map(e=>({at:e.at,status:e.status})),signInState:c.ui.includes('Sign in before using sources'),recordedSourceEvents:c.events.filter(e=>endpoint(e).includes('/ingestion-sources')).length}]));
check('Alice identity200 surrounds Sources visit and succeeds during Retry',nativeSummary.sources.identity.some(i=>i.status===200&&i.userId===2&&i.at<'2026-09-18T01:05:37.553Z')&&nativeSummary['sources-retry'].identity.some(i=>i.status===200&&i.userId===2));
check('Sources failure persists through settled observation and Retry',['sources','sources-settled','sources-retry'].every(n=>nativeSummary[n].signInState));
const legacyObserver=bytes(`${base}/observe-native.js`).toString();
const observedKeep=/\/api\/v1\/(?:chat|chats|characters|notes|flashcards|study[_-]packs|media|rag|knowledge|prompts|prompt_collections|collections|ingestion|jobs|keywords)(?:\/|\?|$)/;
check('Legacy matrix observer excludes source list and capability paths',legacyObserver.includes('collections|ingestion|jobs')&&!observedKeep.test('/api/v1/ingestion-sources')&&!observedKeep.test('/api/v1/ingestion-sources/capabilities'));
for(const p of ['capability258-authenticated.js','capability258-authenticated-reload.js','capability258-sources.js','capability258-sources-settled.js','capability258-sources-retry.js'])bytes(`${base}/pg-multi/${p}`);
const observed=capture('pg-multi','capability258-source-observer-retry');
const correctedHelper=bytes(`${base}/pg-multi/capability258-source-request-observer.js`).toString();
const corrected=observed.events.map(e=>({event:e.event,at:e.at,port:new URL(e.url).port,path:endpoint(e),method:e.method,status:e.status,authorizationPresent:e.authorizationPresent,apiKeyPresent:e.apiKeyPresent,cookiePresent:e.cookiePresent,notAuthenticated:JSON.stringify(e.body||{}).toLowerCase().includes('not authenticated')}));
const initial=corrected.filter(e=>e.event==='request'&&e.port==='18783'&&e.path==='/api/v1/ingestion-sources');
const redirected=corrected.filter(e=>e.event==='request'&&e.port==='18703'&&e.path==='/api/v1/ingestion-sources/');
const redirects=corrected.filter(e=>e.status===307),denied=corrected.filter(e=>e.status===401);
check('Corrected observer captures four authenticated same-origin requests redirected to unauthenticated backend requests',initial.length===4&&initial.every(e=>e.authorizationPresent===true)&&redirects.length===4&&redirected.length===4&&redirected.every(e=>e.authorizationPresent===false)&&denied.length===4);
check('Corrected helper serializes only auth header presence booleans',correctedHelper.includes('authorizationPresent:!!h.authorization')&&correctedHelper.includes("apiKeyPresent:!!h['x-api-key']")&&!correctedHelper.includes('headers:h'));

const runtimes=[];let logProjection=[];let multiSourceRoot='';
for(const cell of ['pg-single','pg-multi']){
 const dir=`.tmp/uat-next-matrix-20260916/targeted-upgrades/${run}/${cell}`,bindingPath=`${dir}/binding.private.json`,bb=bytes(bindingPath,true),binding=JSON.parse(bb);
 const pb=bytes(binding.originalProfile,true),profile=JSON.parse(pb),ib=bytes(path.join(profile.root,'initialized.private.json'),true),hb=bytes(profile.pgReceiptPath,true);
 const sourceRoot=binding.sourceRoot,mb=bytes(path.join(path.dirname(sourceRoot),'preparation',`${cell}-source-manifest.json`)),manifest=JSON.parse(mb);
 check(`${cell} source and original fixture bindings match`,binding.sourceCommit===revision&&manifest.revision===revision&&binding.proof.sourceManifest===sha(mb)&&binding.originalProfileHash===sha(pb)&&binding.originalInitializationHash===sha(ib)&&binding.originalHolderHash===sha(hb));
 const receipts=fs.readdirSync(dir).filter(n=>n.endsWith('.process.private.json')).map(n=>{const p=path.join(dir,n),rb=bytes(p,true),r=JSON.parse(rb);return {path:p,value:r,hash:sha(rb)};});
 const processProof=receipts.map(({path:p,value:r,hash})=>({path:p,pid:r.pid,action:r.action,startedAt:r.startedAt,endedAt:r.endedAt||null,observedSha256:hash,immutableBindingMatch:r.bindingHash===sha(bb)&&r.sourceCommit===revision&&r.sourceRoot===sourceRoot,coversFresh:r.startedAt<fresh[cell].at&&(!r.endedAt||r.endedAt>fresh[cell].observedAt)}));
 check(`${cell} owned API/Next receipts cover fresh capture`,processProof.length===2&&processProof.every(p=>p.immutableBindingMatch&&p.coversFresh));
 if(cell==='pg-multi')check('Owned PG-multi API/Next receipts also cover authenticated capability and corrected redirect capture',processProof.every(p=>p.startedAt<captures.sources.at&&(!p.endedAt||p.endedAt>observed.observedAt)));
 const cap=bytes(path.join(sourceRoot,'apps/packages/ui/src/services/tldw/server-capabilities.ts'));
 check(`${cell} repaired capability source matches prepared manifest`,manifest.files.find(f=>f.path==='apps/packages/ui/src/services/tldw/server-capabilities.ts')?.sha256===sha(cap)&&cap.toString().includes('if (isActiveCookieSessionConfig(config)) return true'));
 runtimes.push({cell,sourceCommit:revision,bindingSha256:sha(bb),sourceManifestSha256:sha(mb),processProof,capabilitySourceSha256:sha(cap)});
 if(cell==='pg-multi'){
  multiSourceRoot=sourceRoot;
  const backend=receipts.find(r=>path.basename(r.path).startsWith('backend-')),lp=backend.path.replace('.process.private.json','.private.log'),lb=bytes(lp,true);
  const lines=lb.toString().replace(/\x1b\[[0-9;]*m/g,'').split(/\r?\n/);
  logProjection=lines.map((s,i)=>({s,i})).filter(({s})=>s.includes('/api/v1/ingestion-sources')&&/2026-09-17 (?:17:59:52|18:05:37|18:05:38|18:08:02|18:08:03)/.test(s)).map(({s,i})=>({line:i+1,localTime:s.match(/\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+/)?.[0],method:s.match(/\b(GET|POST|PUT|DELETE|OPTIONS)\b/)?.[1],path:s.match(/\/api\/v1\/ingestion-sources[^\s"?]*/)?.[0],status:Number(s.match(/HTTP\/[\d.]+"?\s+(\d{3})/)?.[1]||s.match(/(?:Status:|status_code[=:]|status[=:])\s*(\d{3})/i)?.[1]||0)||null}));
  check('Authenticated Sources capability probe returned200 before redirect failure',logProjection.some(e=>e.localTime==='2026-09-17 18:05:37.553'&&e.path==='/api/v1/ingestion-sources/capabilities'&&e.status===200));
  check('Backend log independently records slashless307 followed by trailing-slash401',logProjection.some(e=>e.path==='/api/v1/ingestion-sources'&&e.status===307)&&logProjection.some(e=>e.path==='/api/v1/ingestion-sources/'&&e.status===401));
 }
}
const sourcePaths=['apps/packages/ui/src/services/tldw/domains/collections.ts','tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py','apps/tldw-frontend/next.config.mjs','apps/packages/ui/src/components/ui/state/capability-state.ts'];
const sourceParity=sourcePaths.map(p=>{const old=bytes(oldRun+p),next=bytes(path.join(multiSourceRoot,p));return {path:p,pre258Sha256:sha(old),runtimeSha256:sha(next),same:old.equals(next)};});
check('Sources route/client/rewrite/error-classifier defect ingredients predate258',sourceParity.every(p=>p.same));
for(const p of ['apps/packages/ui/src/services/tldw/TldwApiClient.ts','apps/packages/ui/src/services/background-proxy.ts','apps/packages/ui/src/hooks/useTldwApiClient.tsx','apps/packages/ui/src/hooks/use-ingestion-sources.ts','apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx','apps/packages/ui/src/components/Option/Sources/SourcesAvailabilityGate.tsx','apps/packages/ui/src/components/Common/WorkspaceConnectionGate.tsx','apps/packages/ui/src/services/__tests__/tldw-api-client.ingestion-sources.test.ts','apps/packages/ui/src/hooks/__tests__/use-ingestion-sources.test.tsx'])bytes(path.join(multiSourceRoot,p));
const priorReview=json('.tmp/uat-repairs-231-246/capability258-review/ROUND2-audit.json');
check('Independent258 source/test review was clear',priorReview.verdict==='CLEAR bounded source/test review'&&priorReview.independentTests.passed===25);
bytes(`${out}/audit.mjs`);
const audit={task:'UAT258 native probe acceptance and separate UAT264 Sources redirect diagnosis',at:new Date().toISOString(),verdict:checks.every(c=>c.pass)?'CLEAR bounded258 probe gate; CONFIRMED separate Sources redirect auth loss':'GAPS',checks,summary:{fresh,nativeAuthenticated:nativeSummary,correctedObserver:{at:observed.at,observedAt:observed.observedAt,events:corrected},backendAccessProjection:{timezone:'America/Los_Angeles (UTC-07:00 at observation)',records:logProjection},runtimes,preexistingSourceParity:sourceParity},corrections:['Earlier zero-ingestion-request conclusions from page.__matrixEvents were invalid: its keep regex excludes ingestion-sources. Raw captures remain unchanged; their absence cannot be used as dispatch proof.','The first preflight-readiness hypothesis was not supported. The Sources Sign in state comes from an actual401, and corrected capture shows Authorization on the initiating request.','Fresh anonymous258 captures have a separate comprehensive observer and remain valid. Authenticated capability200 is established by the bound backend log, not the incomplete matrix observer.'],limits:['Native258 acceptance covers no protected capability request/errors in fresh unconfigured contexts in both PG modes, plus successful authenticated Alice capability dispatch on PG-multi. It does not certify Sources listing, source creation, PG-single authenticated capability behavior, all auth variants, or the full48 matrix.','The Sources list is currently broken: cross-port redirect loses Authorization, then backend correctly returns401. No permission/grant/auth policy relaxation is proposed.','The relevant client path, backend route, Next rewrite and classifier are byte-identical before258 and in edfd. This establishes pre-existing code, not an earlier native reproduction of the list failure.','Mutable backend log and process receipts are hashed at read time; only allowlisted access records are projected. No raw log is copied.','No auth token, credential value, raw reasoning or unrelated response body is serialized; corrected observer records header presence booleans only.','Reviewer performed source/evidence/log reads and wrote this packet only. No browser, runtime, model, DB, Git, tracker or product changes were performed.'],inputs};
fs.writeFileSync(`${out}/audit.json`,JSON.stringify(audit,null,2)+'\n');console.log(JSON.stringify({verdict:audit.verdict,checks:checks.length,failed:checks.filter(c=>!c.pass),inputs:inputs.length,auditSha256:sha(fs.readFileSync(`${out}/audit.json`))}));
