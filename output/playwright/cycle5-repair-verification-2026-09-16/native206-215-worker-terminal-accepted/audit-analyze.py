from pathlib import Path
import hashlib,json
root=Path('.tmp/uat198-181-native-20260917'); out=Path('.tmp/uat206-215-native-audit-20260917')
names=['study-pack-job2-repaired-api-result.json','reviewed-worker76778-startup.json','study-pack-job5-list-native-api-result.json','reviewed218-before-restart.json','reviewed218-after-health.json','alice218-pack-submitted.txt','alice218-pack-pending-events.txt','alice218-pack-running.txt','alice218-pack-progress-events.txt','alice218-pack-after-generation.txt','alice218-pack-after-generation-events.txt','native219-failure-source.json']
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(n,j): (out/n).write_text(json.dumps(j,indent=2)+'\n')
write('input-manifest.json',{'inputs':[{'path':str(root/n),'bytes':(root/n).stat().st_size,'sha256':sha((root/n).read_bytes())} for n in names]})
def cli(n):return json.loads((root/n).read_text().split('### Result\n',1)[1].split('\n### Ran',1)[0])
j=cli('alice218-pack-after-generation-events.txt'); ev=[e for e in j['events'] if e.get('at','')>='2026-09-17T08:12:00' and '/study-packs' in e.get('url','')]
responses=[e for e in ev if e['event']=='response'];groups={};transitions=[];prev=None
for e in sorted(responses,key=lambda x:x['at']):
 b=json.loads(e['body']); state=(e['status'],b.get('job',{}).get('status'));groups.setdefault(str(state),[]).append(e['at'])
 if state!=prev:transitions.append({'at':e['at'],'url':e['url'],'status':e['status'],'body':b});prev=state
ms=[json.loads((root/n).read_text()) for n in ['reviewed218-before-restart.json','reviewed218-after-health.json','native219-failure-source.json']]
assert ms[0]['files']==ms[1]['files']==ms[2]['files']
paths=['tldw_Server_API/app/api/v1/endpoints/flashcards.py','tldw_Server_API/app/api/v1/schemas/flashcards.py','tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py','tldw_Server_API/app/services/study_pack_jobs_worker.py']
hashes={f['path']:f['sha256'] for f in ms[0]['files']}
write('receipt.json',{'verdicts':{'215':'native original-job terminal failure gate PASS','206':'native default-worker queued-job completion gate PASS','219':'separate result-display failure remains unresolved by these inputs'},'source':{'revision':ms[0]['revision'],'all_3668_manifest_entries_identical':True,'captures':[m['at'] for m in ms],'relevant_hashes':{p:hashes[p] for p in paths}},'worker':json.loads((root/'reviewed-worker76778-startup.json').read_text()),'ui_submission_requests':[e for e in ev if e['event']=='request' and e.get('method')=='POST'],'ui_status_transition_responses':transitions,'detail_response_groups':{k:{'count':len(v),'first':min(v),'last':max(v)} for k,v in groups.items()},'last_event_capture':j['at'],'last_identity_before_submit':[e for e in j['events'] if e.get('event')=='identity'][-1],'api_job2':json.loads((root/'study-pack-job2-repaired-api-result.json').read_text()),'api_job5_catalogue':json.loads((root/'study-pack-job5-list-native-api-result.json').read_text()),'limitations':['No successful completed-job detail response or generated-pack rendering; no generated-card quality or complete persistence-content claim.','The running-to-queued-to-running transition is recorded; retry reason is not established by this allowlist.','No raw logs, credential files, live process/environment query or new browser/provider actions used for this audit.','Source and startup attribution use retained parent receipts; no independent live process-memory verification.']})
