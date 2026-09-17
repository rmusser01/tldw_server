// TASK13260: Read-only API corroboration using normal owned UAT account login.
import fs from 'node:fs';import path from 'node:path';import {fileURLToPath} from 'node:url';
const packet=path.dirname(fileURLToPath(import.meta.url));const accounts=JSON.parse(fs.readFileSync(path.join(packet,'profiles/tldw-onboarding-uat-fresh-final-20260917-sqlite-multi/credentials.private.json'))).accounts;
const base='http://127.0.0.1:18601/api/v1';const report={at:new Date().toISOString(),kind:'Read-only reciprocal populated-deck/card/job checks; independent password login',checks:[],completed:false};
function check(ok,label){if(!ok)throw Error(label)}
try{for(const who of ['alice','bob']){const a=accounts[who];const r=await fetch(base+'/auth/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams(a),signal:AbortSignal.timeout(10000)});const auth=await r.json();check(r.ok&&typeof auth.access_token==='string','login failed');const headers={Authorization:`Bearer ${auth.access_token}`};
 const get=async(route)=>{const r=await fetch(base+route,{headers,signal:AbortSignal.timeout(10000)});const out={who,route,status:r.status,body:await r.json()};report.checks.push(out);return out};
 const me=await get('/auth/me');check(me.status===200&&me.body.id===(who==='alice'?2:3),'identity mismatch');
 const decks=await get('/flashcards/decks');check(decks.status===200&&decks.body.length===1&&decks.body[0].name===(who==='alice'?'Alice Private Deck ORBIT742':'Bob Private Deck BIRCH913'),'wrong deck catalogue');
 for(const owner of ['alice','bob']){const card=await get('/flashcards/id/'+(owner==='alice'?'81abd812-2c27-47d9-b926-f0134b155b81':'0665ebfb-836d-46c3-b573-f22260ed9731'));check(card.status===(owner===who?200:404),'card ownership boundary failed');}
 const job=await get('/media/ingest/jobs/6');check(job.status===(who==='bob'?200:403),'Bob job boundary failed');
 }report.completed=true;report.completedAt=new Date().toISOString();}catch(e){report.error=e.message;process.exitCode=1}
fs.writeFileSync(path.join(packet,'native/sqlite-multi/isolation-api-supplement.json'),JSON.stringify(report,null,2)+'\n',{mode:0o600});console.log(JSON.stringify({completed:report.completed,checks:report.checks.length,error:report.error}));
