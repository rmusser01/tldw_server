import fs from 'node:fs';
const base='http://127.0.0.1:18501';
const runtime=JSON.parse(fs.readFileSync('/private/tmp/tldw-onboarding-uat-cycle5-multi-20260916/runtime-private.json','utf8'));
const tokens={}, secrets=[], results=[];
const output='/private/tmp/cycle5-multi-native-admin-fixture.json';
function save(){const s=JSON.stringify(results,null,2);for(const v of secrets)if(v&&s.includes(v))throw Error('Sensitive value in evidence');fs.writeFileSync(output,s,{mode:0o600});}
for(const who of ['admin']){const c=runtime.credentials[who];secrets.push(c.password);const r=await fetch(base+'/api/v1/auth/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({username:c.username,password:c.password})});if(!r.ok)throw Error(who+' login status '+r.status);const j=await r.json();tokens[who]=j.access_token;secrets.push(j.access_token,j.refresh_token);results.push({who,action:'real API login',status:r.status,expires_in:j.expires_in});}
async function call(who,method,path,body,headers={},expect){const multipart=body instanceof FormData;const r=await fetch(base+path,{method,headers:{Authorization:'Bearer '+tokens[who],...(body&&!multipart?{'Content-Type':'application/json'}:{}),...headers},...(body?{body:multipart?body:JSON.stringify(body)}:{})});const t=await r.text();let j;try{j=JSON.parse(t)}catch{j=t.slice(0,500)}results.push({who,method,path,status:r.status,...(body&&!multipart?{request:body}:{}),body:j});save();if(expect&&!expect.includes(r.status))throw Error('Unexpected '+who+' '+method+' '+path+' '+r.status+'; stopped, retained evidence');return j;}

await call('admin','POST','/api/v1/media/search',{}, {},[200]);
const form=new FormData();form.append('media_type','document');form.append('perform_analysis','false');form.append('perform_chunking','false');form.append('title','Cycle5 Admin only restore fixture');form.append('files',new Blob(['CYCLE5-ADMIN-RESTORE synthetic only source. The coral cabinet stores 9 shells.'],{type:'text/plain'}),'cycle5-admin-restore.txt');await call('admin','POST','/api/v1/media/add',form,{},[200,201]);console.log(JSON.stringify({output,steps:results.length}));
