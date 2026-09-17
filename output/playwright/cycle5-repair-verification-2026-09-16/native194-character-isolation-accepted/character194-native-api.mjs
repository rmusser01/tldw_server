import fs from 'node:fs';
import assert from 'node:assert/strict';
const profile=JSON.parse(fs.readFileSync('.tmp/fresh-uat-recovery-20260916/pg-multi.profile.private.json'));
const accounts=JSON.parse(fs.readFileSync(profile.credentialsPath)).accounts;
const origin=`http://127.0.0.1:${profile.spec.api}`;
assert.equal(origin,'http://127.0.0.1:18503');
const records=[],tokens=new Map();
const out='.tmp/uat198-181-native-20260917/character194-native-api-result.json';
assert.ok(!fs.existsSync(out));
let passed=false,alice,bob;
const name='UAT194 private character 20260917 0757';
function record(value){
 const serialized=JSON.stringify(value);
 for(const token of tokens.values())assert.ok(!serialized.includes(token));
 for(const a of Object.values(accounts))assert.ok(!serialized.includes(a.password));
 records.push(value);
}
async function req(actor,method,path,body,expected=200){
 const response=await fetch(origin+path,{method,headers:{Authorization:`Bearer ${tokens.get(actor)}`,...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(20000)});
 const value=await response.json();
 record({at:new Date().toISOString(),actor,method,path,request:body,status:response.status,body:path==='/api/v1/auth/me'?{id:value.id,username:value.username}:value});
 assert.equal(response.status,expected,`${actor} ${method} ${path}`);
 return value;
}
try{
 for(const actor of ['Alice','Bob']){
  const account=accounts[actor];
  const r=await fetch(origin+'/api/v1/auth/login',{method:'POST',body:new URLSearchParams({username:account.username,password:account.password}),signal:AbortSignal.timeout(20000)});
  record({at:new Date().toISOString(),actor,method:'POST',path:'/api/v1/auth/login',status:r.status});assert.equal(r.status,200);
  const token=(await r.json()).access_token;assert.equal(typeof token,'string');tokens.set(actor,token);
  assert.equal(Number((await req(actor,'GET','/api/v1/auth/me')).id),actor==='Alice'?2:3);
 }
 alice=await req('Alice','POST','/api/v1/characters/',{name,description:'Synthetic Alice-only UAT194 violet compass.',first_message:'Alice fixture greeting.'},201);
 assert.equal(alice.version,1);
 await req('Bob','GET',`/api/v1/characters/${alice.id}`,undefined,404);
 assert.ok(!(await req('Bob','GET','/api/v1/characters/')).some(x=>x.id===alice.id));
 assert.equal((await req('Bob','GET',`/api/v1/characters/query?query=${encodeURIComponent(name)}`)).total,0);
 assert.deepEqual(await req('Bob','GET',`/api/v1/characters/search/?query=${encodeURIComponent(name)}`),[]);
 await req('Bob','PUT',`/api/v1/characters/${alice.id}?expected_version=1`,{description:'Unauthorized synthetic attempt'},404);
 await req('Bob','DELETE',`/api/v1/characters/${alice.id}?expected_version=1`,undefined,404);
 const unchanged=await req('Alice','GET',`/api/v1/characters/${alice.id}`);assert.equal(unchanged.version,1);assert.equal(unchanged.description,alice.description);
 alice=await req('Alice','PUT',`/api/v1/characters/${alice.id}?expected_version=1`,{description:'Synthetic Alice-only UAT194 violet compass updated.'});assert.equal(alice.version,2);
 bob=await req('Bob','POST','/api/v1/characters/',{name,description:'Synthetic Bob-only UAT194 amber map.',first_message:'Bob fixture greeting.'},201);assert.notEqual(bob.id,alice.id);
 await req('Alice','GET',`/api/v1/characters/${bob.id}`,undefined,404);
 for(const [actor,own,foreign] of [['Alice',alice,bob],['Bob',bob,alice]]){
  const listed=await req(actor,'GET','/api/v1/characters/');assert.ok(listed.some(x=>x.id===own.id));assert.ok(!listed.some(x=>x.id===foreign.id));
  const queried=await req(actor,'GET',`/api/v1/characters/query?query=${encodeURIComponent(name)}`);assert.equal(queried.total,1);assert.equal(queried.items[0].id,own.id);
 }
 await req('Alice','DELETE',`/api/v1/characters/${alice.id}?expected_version=2`);
 await req('Bob','POST',`/api/v1/characters/${alice.id}/restore?expected_version=3`,undefined,404);
 assert.equal((await req('Bob','GET',`/api/v1/characters/query?query=${encodeURIComponent(name)}&include_deleted=true`)).total,1);
 alice=await req('Alice','POST',`/api/v1/characters/${alice.id}/restore?expected_version=3`);assert.equal(alice.version,4);
 const restored=await req('Alice','GET',`/api/v1/characters/${alice.id}`);assert.equal(restored.version,4);assert.equal(restored.description,'Synthetic Alice-only UAT194 violet compass updated.');
 assert.equal((await req('Bob','GET',`/api/v1/characters/${bob.id}`)).version,1);
 passed=true;
}finally{
 for(const actor of tokens.keys()){
  try{await req(actor,'POST','/api/v1/auth/logout',{all_devices:false});}catch(e){record({actor,cleanupError:e.message});passed=false;}
 }
 tokens.clear();
 fs.writeFileSync(out,JSON.stringify({at:new Date().toISOString(),passed,scope:'Running PostgreSQL API; fresh synthetic Alice/Bob sessions, no browser credential access. Test-created characters only. Existing native service role remains privileged; this verifies application ownership.',aliceId:alice?.id,bobId:bob?.id,records},null,2)+'\n',{mode:0o600,flag:'wx'});
}
console.log(JSON.stringify({passed,aliceId:alice.id,bobId:bob.id,requests:records.length}));
