// TASK13260: API corroboration of native, UI-created fixtures. No browser tokens or DB access.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const packet=path.dirname(fileURLToPath(import.meta.url));
const root=path.join(packet,'profiles/tldw-onboarding-uat-fresh-final-20260917-sqlite-multi');
const accounts=JSON.parse(fs.readFileSync(path.join(root,'credentials.private.json'),'utf8')).accounts;
const base='http://127.0.0.1:18601/api/v1';
const report={startedAt:new Date().toISOString(),kind:'API corroboration; all fixtures created through native UI; independent normal password login',checks:[]};
const sessions={};
async function request(who,route,{method='GET',body,version}={}){
 const headers={Authorization:`Bearer ${sessions[who]}`};
 if(body!==undefined)headers['Content-Type']='application/json';
 if(version!==undefined)headers['expected-version']=String(version);
 const r=await fetch(base+route,{method,headers,body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(15000)});
 let data;try{data=await r.json()}catch{data={unparsed:true}}
 const out={at:new Date().toISOString(),who,method,route,status:r.status,body:data};report.checks.push(out);return out;
}
function assert(ok,label){if(!ok)throw Error(label)}
try{
 for(const who of ['alice','bob']){
  const a=accounts[who];const r=await fetch(base+'/auth/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({username:a.username,password:a.password}),signal:AbortSignal.timeout(15000)});
  const b=await r.json();assert(r.ok&&typeof b.access_token==='string','normal login failed');sessions[who]=b.access_token;
  const me=await fetch(base+'/auth/me',{headers:{Authorization:`Bearer ${sessions[who]}`},signal:AbortSignal.timeout(15000)});const m=await me.json();assert(me.ok&&m.id===(who==='alice'?2:3),'identity mismatch');report.checks.push({who,kind:'normal-login-identity',status:r.status,id:m.id,username:m.username});
 }
 const noteIds={alice:'4b06d0b4-51a5-4857-8d83-4082df846767',bob:'b6ed11ef-8ac1-4b92-82c2-c5fd1e3bab4a'};
 const chatIds={alice:'af70ef91-059c-44bb-b106-8704f61e8820',bob:'e24e6232-aae1-472e-98c5-3c29906ababf'};
 for(const owner of ['alice','bob']){
  const other=owner==='alice'?'bob':'alice',route='/notes/'+noteIds[owner];
  const own=await request(owner,route);assert(own.status===200,'own Note read failed');
  const foreign=await request(other,route);assert([403,404].includes(foreign.status),'foreign Note read not denied');
  const edited=own.body.content+'\n\n'+owner.toUpperCase()+' OWN API WRITE CONTROL';
  const write=await request(owner,route,{method:'PUT',body:{content:edited},version:own.body.version});assert(write.status===200&&write.body.content===edited,'valid owner Note write failed');
  const blocked=await request(other,route,{method:'PUT',body:{content:edited+' FOREIGN WRITE SHOULD BE REJECTED'},version:write.body.version});
  const reread=await request(owner,route);const restored=await request(owner,route,{method:'PUT',body:{content:own.body.content},version:reread.body.version});
  assert([403,404].includes(blocked.status),'valid foreign Note write not denied');assert(reread.body.content===edited,'foreign write changed owner Note');assert(restored.status===200&&restored.body.content===own.body.content,'owner Note restoration failed');
  for(const suffix of ['', '/messages?scope_type=global&include_deleted=false&limit=200']){
   const cr='/chats/'+chatIds[owner]+suffix;const a=await request(owner,cr);const b=await request(other,cr);assert(a.status===200&&[403,404].includes(b.status),'reciprocal Chat boundary failed');
  }
 }
 for(const who of ['alice','bob']){
  await request(who,'/media/1');
  await request(who,'/media/ingest/jobs/1');
  await request(who,'/flashcards/id/56320d51-cc2f-4049-baed-65a888230112');
  await request(who,'/flashcards/decks');
  await request(who,'/chat/conversations?order_by=recency&limit=50&keywords=__knowledge_QA__');
 }
 report.completedAt=new Date().toISOString();report.completed=true;
}catch(e){report.completed=false;report.failure=String(e.message).replace(/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,'[REDACTED]');process.exitCode=1}
fs.writeFileSync(path.join(packet,'native/sqlite-multi/isolation-api-corroboration.json'),JSON.stringify(report,null,2)+'\n',{mode:0o600});
console.log(JSON.stringify({completed:report.completed,checks:report.checks.length,failure:report.failure}));
