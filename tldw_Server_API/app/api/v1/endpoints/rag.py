# Server_API/app/api/v1/endpoints/rag.py
# Description: This code provides a FastAPI endpoint for RAG interaction
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
from tldw_Server_API.app.core.RAG.RAG_Library_2 import (
    search_functions
)
#
# DB Mgmt
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
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
