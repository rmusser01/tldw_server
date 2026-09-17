from pathlib import Path
import hashlib,json,datetime
p=Path('.tmp/uat198-181-native-20260917');out=Path('.tmp/uat197-204-216-219-220-native-audit-20260917');prior=Path('output/playwright/cycle5-repair-verification-2026-09-16/native206-215-worker-terminal-accepted')
names=['study-pack-job5-complete-readback-result.json','study-pack-job5-bounded-continuation-result.json','reviewed220-before-restart.json','reviewed220-after-health.json','overlap82583-health.json','known-api-drain-completion.json','alice219-deck-settled.txt','alice220-study-pack-review-open.txt','alice220-assistant-open.txt','alice220-assistant-settled.txt','alice220-assistant-events.txt','alice220-deep-dive-open.txt','alice220-deep-dive-settled.txt','alice220-deep-dive-events.txt']
prior_names=['reviewed-worker76778-startup.json','study-pack-job5-list-native-api-result.json','audit-receipt.json']
inputs=[p/n for n in names]+[prior/n for n in prior_names]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def save(n,j): (out/n).write_text(json.dumps(j,indent=2)+'\n')
save('input-manifest.json',{'inputs':[{'path':str(x),'bytes':x.stat().st_size,'sha256':sha(x)} for x in inputs]})
first=json.loads((p/names[0]).read_text());follow=json.loads((p/names[1]).read_text())
assert first['passed'] is False and follow['passed'] is True
records=follow['records']
def records_for(actor,path):return [r for r in records if r['actor']==actor and r['path']==path]
def body(actor,path):return records_for(actor,path)[0]['body']
assert body('Alice','/api/v1/auth/me')['id']==2 and body('Bob','/api/v1/auth/me')['id']==3
job=body('Alice','/api/v1/flashcards/study-packs/jobs/5');pack=body('Alice','/api/v1/flashcards/study-packs/2')
assert job['job']['id']==5 and job['job']['status']=='completed' and job['error'] is None and job['study_pack']==pack
assert pack['id']==2 and pack['deck_id']==10 and pack['client_id']=='2' and pack['version']==1
assert all(r['body']==job for r in records_for('Alice','/api/v1/flashcards/study-packs/jobs/5'))
deck=next(d for d in body('Alice','/api/v1/flashcards/decks') if d['id']==10);assert deck['client_id']=='2'
cards=body('Alice','/api/v1/flashcards?deck_id=10');assert cards['count']==cards['total']==3 and not cards['has_more']
assert len({x['uuid'] for x in cards['items']})==3 and all(c['client_id']=='2' and c['deck_id']==10 for c in cards['items'])
source='b83dca90-fab0-4c6f-8c0f-6f1e93dfffc8'; details=[]
for card in cards['items']:
 a=body('Alice',f"/api/v1/flashcards/{card['uuid']}/assistant")
 assert a['study_pack']==pack and len(a['citations'])==1 and a['primary_citation']==a['citations'][0]
 cite=a['primary_citation'];assert cite['flashcard_uuid']==card['uuid'] and cite['source_id']==source and cite['client_id']=='2'
 for obj in [pack,cite]:
  for field in ['created_at','last_modified']:
   assert isinstance(obj[field],str);assert datetime.datetime.fromisoformat(obj[field]).tzinfo is not None
 details.append({'card':card,'citation':cite,'membership_pack_id':a['study_pack']['id'],'thread':a['thread'],'message_count':len(a['messages'])})
assert records_for('Bob','/api/v1/flashcards/study-packs/jobs/5')[0]['status']==404
assert body('Bob','/api/v1/flashcards?deck_id=10')['total']==0
assert records_for('Bob',f"/api/v1/flashcards/{cards['items'][0]['uuid']}/assistant")[0]['status']==404
leak=next(r for r in first['records'] if r['actor']=='Bob' and r['path']=='/api/v1/flashcards/study-packs/2')
assert leak['status']==200 and leak['body']==pack
note=body('Alice',f'/api/v1/notes/{source}'); original=body('Alice','/api/v1/flashcards/id/37b10bd7-edf4-4f35-83c1-1490115d8c55')
assert note['version']==1 and note['client_id']=='2' and note['content']==pack['source_bundle_json']['items'][0]['evidence_text']
assert original['version']==2 and original['repetitions']==1 and original['client_id']=='2'
ms=[json.loads((p/n).read_text()) for n in ['reviewed220-before-restart.json','reviewed220-after-health.json']];assert ms[0]['files']==ms[1]['files'];hashes={x['path']:x['sha256'] for x in ms[0]['files']}
assert hashes['tldw_Server_API/app/api/v1/schemas/study_packs.py']=='96843579f5f1c87a855dcb7ab4220bd8114c44ab5ce6ae60188c7add0705349a'
def cli(n):return json.loads((p/n).read_text().split('### Result\n',1)[1].split('\n### Ran',1)[0])
e=cli('alice220-assistant-events.txt'); relevant=[r for r in e['events'] if r.get('event')=='response' and r.get('at','')>'2026-09-17T08:42' and '/assistant' in r.get('url','')]
assert len(relevant)==1 and relevant[0]['status']==200
assert json.loads(relevant[0]['body'])['primary_citation']['citation_text']=='The Citrine study marker is amber.'
ui=(p/'alice220-assistant-settled.txt').read_text();assert '3 cards remaining, 0 reviewed' in ui and 'The Citrine study marker is amber.' in ui
assert 'HTTP status: 404 Not Found' in (p/'alice220-deep-dive-settled.txt').read_text()
save('receipt.json',{'native_gates':{str(n):'PASS within task-specific scope; independent automation previously retained' for n in [197,204,216,219,220]},'original_failed_readback':{'passed':False,'records':len(first['records']),'failure_boundary':leak},'bounded_continuation':{'passed':True,'records':len(records),'explicitly_excludes':'UAT222 foreign pack metadata defect'},'same_original_job':job,'deck':deck,'cards_and_citations':details,'negative_controls':[r for r in records if r['actor']=='Bob' and r['method']=='GET'],'source_note_observed':{'id':source,'version':note['version'],'client_id':note['client_id'],'source_content_matches_bundle':True},'original_card_observed':{k:original[k] for k in ['uuid','version','repetitions','client_id','deleted']},'ui_assistant_response':relevant[0],'source':{'revision':ms[0]['revision'],'captures':[m['at'] for m in ms],'same_3668_source_entries':True,'hashes':{k:hashes[k] for k in ['tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py','tldw_Server_API/app/api/v1/schemas/study_packs.py','tldw_Server_API/app/services/study_pack_jobs_worker.py','tldw_Server_API/app/api/v1/endpoints/flashcards.py']}},'runtime':json.loads((p/'overlap82583-health.json').read_text()),'later_drain':json.loads((p/'known-api-drain-completion.json').read_text()),'limits':['UAT222 foreign pack metadata200 remains a proven failure; no complete Study Pack ownership acceptance.','UAT223 actual Deep dive to source navigation404 remains separate; citation rendering is accepted, source navigation is not.','GET assistant creates/returns empty assistant threads; do not claim no database side effects. No generation, rating, source/card/job mutation was issued by this audit.','Native cold-cache state is not proven; prior official fixture cold/warm controls support204.','No205 complete UI creation/navigation acceptance or full UAT matrix claim.']})
print({'input_count':len(inputs),'records':len(first['records'])+len(records),'cards':len(details),'native_gates':[197,204,216,219,220],'separate_failures':[222,223]})
