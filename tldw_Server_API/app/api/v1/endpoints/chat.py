# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
# FIXME
#
# Imports
#
# 3rd-party imports
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
    UploadFile
)
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import redis
import requests
# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger
from starlette.responses import JSONResponse
#
# Local Imports
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    get_character_names,
    get_conversation_name,
    alert_token_budget_exceeded,
)
#
# DB Mgmt
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_Server_API.app.core.DB_Management.DB_Dependency import get_db_manager
#from tldw_Server_API.app.core.DB_Management.DB_Manager import DBManager
#
#
#######################################################################################################################
#
# Functions:


# All functions below are endpoints callable via HTTP requests and the corresponding code executed as a result of it.
#
# The router is a FastAPI object that allows us to define multiple endpoints under a single prefix.
# Create a new router instance
router = APIRouter()


#
# End of media.py
#######################################################################################################################
