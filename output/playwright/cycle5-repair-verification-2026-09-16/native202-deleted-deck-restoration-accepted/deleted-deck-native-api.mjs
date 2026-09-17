import fs from 'node:fs';
import assert from 'node:assert/strict';
const profile=JSON.parse(fs.readFileSync('.tmp/fresh-uat-recovery-20260916/pg-multi.profile.private.json'));
const account=JSON.parse(fs.readFileSync(profile.credentialsPath)).accounts.Bob;
const origin=`http://127.0.0.1:${profile.spec.api}`;
assert.equal(origin,'http://127.0.0.1:18503');
const records=[];let token;let passed=false;
const output='.tmp/uat198-181-native-20260917/deleted-deck-native-api-result.json';
async function request(method,path,body){const response=await fetch(origin+path,{method,headers:{Authorization:`Bearer ${token}`,...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(20000)});let value;try{value=await response.json()}catch{}records.push({at:new Date().toISOString(),method,path,status:response.status,body:path==='/api/v1/auth/me'?{id:value?.id,username:value?.username}:value});assert.ok(response.ok,`${method} ${path} status ${response.status}`);return value;}
try{
 const login=await fetch(origin+'/api/v1/auth/login',{method:'POST',body:new URLSearchParams({username:account.username,password:account.password}),signal:AbortSignal.timeout(20000)});
 records.push({at:new Date().toISOString(),method:'POST',path:'/api/v1/auth/login',status:login.status});assert.equal(login.status,200);
 const session=await login.json();token=session.access_token;assert.equal(typeof token,'string');
 const identity=await request('GET','/api/v1/auth/me');assert.equal(Number(identity.id),3);
 const baseline=await request('GET','/api/v1/flashcards/decks');assert.ok(baseline.every(d=>String(d.client_id)==='3'));
 const name='Bob UAT202 disposable restore 20260917T0615';assert.ok(!baseline.some(d=>d.name===name));
 const created=await request('POST','/api/v1/flashcards/decks',{name,description:'Owned empty UAT202 restore control',visibility:'private'});assert.equal(String(created.client_id),'3');assert.equal(created.version,1);
 await request('DELETE',`/api/v1/flashcards/decks/${created.id}?expected_version=${created.version}`);
 const deleted=await request('GET','/api/v1/flashcards/decks?include_deleted=true');const tombstone=deleted.find(d=>d.id===created.id);assert.equal(tombstone.deleted,true);assert.equal(tombstone.version,2);
 const restored=await request('POST','/api/v1/flashcards/decks',{name,description:'Owned empty UAT202 restore control',visibility:'private'});assert.equal(restored.id,created.id);assert.equal(restored.version,3);assert.equal(restored.deleted,false);assert.equal(String(restored.client_id),'3');
 const final=await request('GET','/api/v1/flashcards/decks');assert.deepEqual(final.find(d=>d.id===created.id),restored);assert.deepEqual(final.filter(d=>d.id!==created.id),baseline);
 passed=true;
}finally{
 if(token){await request('POST','/api/v1/auth/logout',{all_devices:false});token=undefined;}
 fs.writeFileSync(output,JSON.stringify({at:new Date().toISOString(),scope:'native API; fresh synthetic Bob login, no browser session access; own new empty deck only',passed,records},null,2)+'\n',{mode:0o600});
}
console.log(JSON.stringify({passed,requests:records.length,result:output}));
