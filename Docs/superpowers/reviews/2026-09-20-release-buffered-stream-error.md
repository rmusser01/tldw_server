# Buffered character stream error boundary

Tracking: TASK-13263. CodeQL alert2693 on candidate PR2972.

The authenticated completion route returns a StreamingResponse for offline simulation and buffered provider replies. Its generator previously serialized str(exception) after response creation, beyond the route HTTP exception mapper. Caught internal faults could disclose paths or other sensitive exception details. No ordinary request was shown to reliably induce these faults.

The helper now uses the adjacent lazy-stream handler's constant public error and logs only exception class. SSE error-string shape, HTTP response, headers, success role/content/stop frames, credential lifecycle and terminal DONE remain unchanged. Two direct callers were independently traced before implementation, and a fresh read-only reviewer found no surviving bypass or regression in the final patch.

Regression evidence: initial provider fallback run reproduced two failures (RuntimeError and ValueError leak) and passed the legitimate control. Expanded tests cover both offline and buffered-provider modes, faults before a header and during content serialization, absence of the private sentinel, safe terminal error/DONE and normal successful content. Full owning error-mapping module: **129 passed**. Bandit production module: zero findings. Ruff: five current diagnostics equal five exact-HEAD baseline diagnostics, zero new; unchanged legacy warnings are not claimed clean. git diff --check passes. No whole repository or deployed validation claim.

Local artifacts: /tmp/qodo-buffered-stream-red.log, /tmp/qodo-buffered-stream-green.log, /tmp/qodo-buffered-stream-bandit.json.
