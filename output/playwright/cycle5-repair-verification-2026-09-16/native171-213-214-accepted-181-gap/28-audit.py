from pathlib import Path
import json, hashlib, re
from urllib.parse import urlsplit
ROOT = Path.cwd()
BASE = ROOT / '.tmp/uat198-181-native-20260917'
META = ROOT / '.tmp/uat181-full-native-20260917'
OUT = ROOT / '.tmp/uat171-181-213-214-native-audit-20260917'
inputs = []
def read(path):
    data = path.read_bytes()
    inputs.append({'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)})
    return data.decode()
def obj(path): return json.loads(read(path))
def events(path):
    text=read(path)
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
def rows(manifest): return {r['path']:r['sha256'] for r in manifest['files']}
def body(event):
    value=event.get('body')
    return json.loads(value) if isinstance(value,str) else value
old_before=obj(BASE/'reviewed218-before-restart.json');old_after=obj(BASE/'reviewed218-after-health.json');old_health=obj(BASE/'reviewed218-replacement-health.json')
new_before=obj(BASE/'reviewed220-before-restart.json');new_after=obj(BASE/'reviewed220-after-health.json')
assert rows(old_before)==rows(old_after) and rows(new_before)==rows(new_after)
assert len(rows(new_after))==3668 and old_health['newPid']==76778
changed=[key for key,value in rows(old_after).items() if value!=rows(new_after).get(key)]
assert changed==['tldw_Server_API/app/api/v1/schemas/study_packs.py']
review=obj(ROOT/'.tmp/uat213-214-root-independent-20260917/source-manifest.json')
read(ROOT/'.tmp/uat213-214-root-independent-20260917/REVIEW213-214.md')
for f in review['files']:
    if '/app/' in f['path']:
        assert rows(old_after)[f['path']]==f['sha256']==rows(new_after)[f['path']]
stop=obj(BASE/'overlap76778-to220-stop.json');health=obj(BASE/'overlap82583-health.json');drain=obj(BASE/'known-api-drain-completion.json')
assert stop['oldPid']==health['oldPid']==76778 and health['newPid']==82583
assert health['rows'][0]['status']==200 and health['rows'][0]['oldAlive'] and health['rows'][0]['newAlive']
assert drain['knownUatApiAlive']=={'56113':False,'76778':False,'82583':True}
metas=[obj(META/(name+'.json')) for name in ['warmed76778-before220-restart','overlap82583-after-health','after82583-study-citations']]
assert len({m['targetContentDatabaseSha256'] for m in metas})==1
assert all(not m['locks'] and all(s['state']=='idle' for s in m['sessions']) and m['transactionReadOnlyVerified'] for m in metas)
notes=events(BASE/'bob181-overlap-notes-events.txt');chat=events(BASE/'bob213-214-after-send-events.txt');buddy=events(BASE/'bob181-buddy-events.txt');deck=events(BASE/'alice219-deck-events.txt')
notes_ui=read(BASE/'bob181-overlap-notes-settled.txt');chat_ui=read(BASE/'bob213-214-after-send.txt');read(BASE/'bob181-buddy-settled.txt');read(BASE/'alice219-deck-settled.txt')
latest_identity=next(e for e in reversed(notes['events']) if e['event']=='identity')
assert latest_identity['status']==200 and latest_identity['identity']['id']==3
notes_after=[e for e in notes['events'] if e['event']=='response' and e['at']>health['rows'][0]['at'] and urlsplit(e['url']).path.startswith('/api/v1/notes')]
assert {urlsplit(e['url']).path for e in notes_after}=={'/api/v1/notes/','/api/v1/notes/keywords/','/api/v1/notes/collections'}
assert all(e['status']==200 for e in notes_after) and 'Showing 1-3 of 3' in notes_ui
chat_responses=[e for e in chat['events'] if e['event']=='response' and e['at']>'2026-09-17T08:40:20Z']
find=lambda suffix:next(e for e in chat_responses if suffix in e['url'])
user=find('/messages?');completion=find('/complete-v2?');persist=find('/completions/persist?')
assert (user['status'],completion['status'],persist['status'])==(201,200,200)
u=body(user);p=body(persist);assert p['saved']
canonical=next(e for e in reversed(notes['events']) if e['event']=='response' and '/messages?' in e['url'] and e['status']==200)
messages=body(canonical)['messages'];assistant=next(m for m in messages if m['id']==p['assistant_message_id'])
assert assistant['parent_message_id']==u['id'] and assistant['content'].endswith('READY') and re.search(r'paragraph[^\n]*: READY\s*$',chat_ui,re.M)
known=obj(BASE/'native213-214-known-error-check.json');assert known['runtimePid']==76778 and set(known['counts'].values())=={0}
start=old_health['startedAt'];finish=stop['rows'][0]['at']
def direct_card_deck(e):
    return e.get('event')=='response' and urlsplit(e.get('url','')).path in ['/api/v1/flashcards','/api/v1/flashcards/decks']
warm=[e for capture in [notes,buddy,deck] for e in capture['events'] if direct_card_deck(e) and start<=e['at']<finish]
assert not warm
later=[e for e in deck['events'] if direct_card_deck(e) and e['at']>finish];assert later and all(e['status']==200 for e in later)
startup_errors=[e for e in notes['events'] if e['event']=='response' and stop['rows'][0]['at']<e['at']<health['rows'][0]['at'] and e['status']>=500]
assert len(startup_errors)==1 and '/research-runs' in startup_errors[0]['url']
for task in ['108','118','152','153']:
    matches=list((ROOT/'backlog/tasks').glob('task-13260.'+task+' - *.md'));assert len(matches)==1;read(matches[0])
summary={
 'disposition':{'UAT171':'native AC3 supported','UAT181':'AC3 remains open: warmed direct Flashcards/deck reads before replacement not evidenced','UAT213':'bounded original-scenario native acceptance supported','UAT214':'bounded original-scenario native acceptance supported'},
 'source':{'old_revision':old_after['revision'],'new_revision':new_after['revision'],'backend_files_each':3668,'each_start_freeze_equal':True,'only_changed_backend_path':changed,'reviewed213214_production_hashes_match_both':True},
 'replacement':{'stop':stop,'health':health,'drain':drain},
 'metadata':[{'observedAt':m['observedAt'],'sessions':len(m['sessions']),'states':sorted({s['state'] for s in m['sessions']}),'tableLocks':len(m['locks']),'relationScope':m['relationScope']} for m in metas],
 'notes':{'identity_id':3,'post_health_responses':[{'at':e['at'],'path':urlsplit(e['url']).path,'status':e['status']} for e in notes_after],'visible_own_notes':True},
 'chat':{'conversation_id':u['conversation_id'],'user_id':u['id'],'user_save_at':user['at'],'completion_status_at':completion['at'],'persist_at':persist['at'],'assistant_id':assistant['id'],'parent_link_matches':True,'canonical_get_at':canonical['at'],'canonical_row_count':len(messages),'visible_answer':'READY','stored_content_includes_model_markup':True,'sse_response_body_available':completion.get('body') is not None,'known_fingerprint_receipt':known['counts']},
 'gap':{'warmed_direct_card_deck_responses':len(warm),'first_post_replacement_card_deck_response':min(e['at'] for e in later),'buddy_receipt_has_only_persona_reads_in_actual_open_window':True},
 'startup_transient':[{'at':e['at'],'path':urlsplit(e['url']).path,'status':e['status']} for e in startup_errors]
}
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'input-manifest.json').write_text(json.dumps({'files':inputs},indent=2)+'\n')
print(json.dumps({'inputs':len(inputs),'disposition':summary['disposition'],'source_files':3668,'metadata_sessions':[len(m['sessions']) for m in metas],'notes_after_health':len(notes_after),'canonical_parent_matches':True,'warm_direct_reads':len(warm)},indent=2))
