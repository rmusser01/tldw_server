import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = root / '.tmp/uat-repairs-231-246/analytics240'
paths = [
    'tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py',
    'tldw_Server_API/tests/DB_Management/test_flashcard_analytics_backends.py',
    'tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py',
    'tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py',
]
summary = []
for index, path in enumerate(paths):
    baseline = subprocess.check_output(['git', 'show', f'HEAD:{path}'])
    current = Path(path).read_bytes()
    diagnostics = {}
    for version, content in [('baseline', baseline), ('current', current)]:
        compile(content, path, 'exec')
        lint = subprocess.run([sys.executable, '-m', 'ruff', 'check', '--output-format', 'json', '--stdin-filename', path, '-'], input=content, capture_output=True)
        lint_data = json.loads(lint.stdout)
        args = [sys.executable, '-m', 'bandit', '-q', '-f', 'json']
        if index:
            args += ['-s', 'B101']
        security = subprocess.run([*args, '-'], input=content, capture_output=True)
        security_data = json.loads(security.stdout)
        (out / f'{index}-{version}-ruff.json').write_text(json.dumps(lint_data, indent=2)+'\n')
        (out / f'{index}-{version}-bandit.json').write_text(json.dumps(security_data, indent=2)+'\n')
        diagnostics[version] = {
            'ruff': sorted((d['code'], d['message']) for d in lint_data),
            'bandit': sorted((d['test_id'], d['issue_text'], d['issue_severity'], d['issue_confidence']) for d in security_data['results']),
            'parseErrors': security_data['errors'],
        }
    before, after = diagnostics['baseline'], diagnostics['current']
    summary.append({'path':path,'sha256':hashlib.sha256(current).hexdigest(),'baselineRuff':len(before['ruff']),'currentRuff':len(after['ruff']),'ruffIdentical':before['ruff']==after['ruff'],'baselineBandit':len(before['bandit']),'currentBandit':len(after['bandit']),'banditIdentical':before['bandit']==after['bandit'],'parseErrors':after['parseErrors'],'compiled':True})
(out/'static-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if any(not x['ruffIdentical'] or not x['banditIdentical'] or x['parseErrors'] for x in summary):
    raise SystemExit(1)
