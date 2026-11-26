"""TASK-13950.25 PR H step 1: #3538 added two executor locals, client_moonshot_history and capture_moonshot_reasoning.
PR H's lifted save_callback and tracked_streaming_generator read them, so they become call-state fields (cst.<name>) first.
A pure in-place rewrite: nothing moves.
"""

MANIFEST = {'base': '164af030cb', 'file': 'tldw_Server_API/app/core/Chat/chat_service.py', 'func': 'execute_streaming_call', 'pkg': '', 'ctx': [['cst', 'StreamCall']], 'ctx_import': ['from tldw_Server_API.app.core.Chat.call_state import StreamCall'], 'facade': 'cs', 'facade_stdlib': ['time'], 'task': 'TASK-13950.25', 'rewrite': {'client_moonshot_history': 'cst.client_moonshot_history', 'capture_moonshot_reasoning': 'cst.capture_moonshot_reasoning'}}
