"""TASK-13950.25 PR J: build_context_and_messages relocates into history_context (verbatim, chat_service globals read as cs.*), and two zero-reference helpers are dropped.
chat_service re-exports build_context_and_messages (chat.py reads it as a module attribute; four tests patch it there).
"""

MANIFEST = {'base': 'ac9a6d2593', 'file': 'tldw_Server_API/app/core/Chat/chat_service.py', 'pkg': '', 'facade': 'cs', 'facade_stdlib': ['time'], 'task': 'TASK-13950.25', 'relocate': [{'name': 'build_context_and_messages', 'module': 'history_context'}], 'drop': [{'module': 'chat_service', 'def': '_apply_redaction_to_content'}, {'module': 'chat_service', 'def': '_resolve_run_first_completion_outcome'}]}
