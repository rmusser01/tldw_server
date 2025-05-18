# tldw_Server_API/app/core/RateLimit.py
# Description: Rate limiting library for FastAPI endpoints
#
# Imports
import time, json, redis.asyncio as redis
#
# 3rd-party Libraries
from fastapi import Request, HTTPException, status
#
# Local Imports
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.config import settings
#
#######################################################################################################################
#
# Functions:

# --- Configuration (using imported settings dictionary) ---
_R = redis.from_url(settings.get("REDIS_URL", "redis://localhost:6379/0"))

# --- simple sliding-window counter ---------------------------------------- #
RATE   = 30                    # requests
WINDOW = 60                    # seconds
TOKENS_DAILY = 100_000         # model tokens per user per UTC day

async def ratelimit_dependency(request: Request):
    user: User = request.state.user                # populated by Auth middleware
    now = int(time.time())
    pipe = _R.pipeline()

    # per-minute request count
    req_key = f"rl:req:{user.id}:{now//WINDOW}"
    pipe.incr(req_key, 1)
    pipe.expire(req_key, WINDOW * 2)

    # daily token budget (client sends X-Token-Usage header)
    tokens_used = int(request.headers.get("X-Token-Usage", "0"))
    day_key = f"rl:tok:{user.id}:{time.strftime('%Y%m%d', time.gmtime())}"
    if tokens_used:
        pipe.incr(day_key, tokens_used)
    pipe.expire(day_key, 86400 * 2)

    req_ct, *_ = await pipe.execute()

    if req_ct > RATE:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="rate-limit exceeded")

    if tokens_used:
        curr = await _R.get(day_key) or b"0"
        if int(curr) > TOKENS_DAILY:
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED,
                                detail="daily token quota exhausted")

#
# End of Rate_Limit.py
#######################################################################################################################
