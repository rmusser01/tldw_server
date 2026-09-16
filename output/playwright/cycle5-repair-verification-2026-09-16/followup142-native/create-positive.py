import json
import os
from dataclasses import asdict
from pathlib import Path
from loguru import logger

logger.remove()
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

profile_root = Path(os.environ['UAT_OWNED_PROFILE_ROOT']).resolve()
db = CollectionsDatabase(user_id='1')
actual_path = Path(db._backend.config.sqlite_path).resolve()
if not actual_path.is_relative_to(profile_root):
    raise RuntimeError('Notification fixture refused: database is outside owned profile')
row = db.create_user_notification(kind='reminder_due', title='UAT142 current authority View', message='Synthetic notification for current-authority and pending View acceptance.', severity='info', link_url='/prompts', dedupe_key='uat142-positive-20260916')
print(json.dumps({'notification': asdict(row), 'database': str(actual_path), 'fixture_only': True}))
