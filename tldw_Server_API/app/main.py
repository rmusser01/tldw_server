# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
#
# 3rd-party Libraries
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#
# Local Imports
from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#
########################################################################################################################
#
# Functions:

app = FastAPI(
    title="tldw API",
    version="0.0.1",
    description="Version 0.0.1: Smooth Slide | FastAPI Backend for the tldw project"
)

# FIXME - CORS
# # -- If you have any global middleware, add it here --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the tldw API; If you're seeing this, the server is running!"}

# Router for media endpoints/media file handling
app.include_router(media_router, prefix="/api/v1/media", tags=["media"])

# Router for trash endpoints - deletion of media items / trash file handling (FIXME: Secure delete vs lag on delete?)
#app.include_router(trash_router, prefix="/api/v1/trash", tags=["trash"])

# Router for authentication endpoint
#app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
# The docs at http://localhost:8000/docs will show an “Authorize” button. You can log in by calling POST /api/v1/auth/login with a form that includes username and password. The docs interface is automatically aware because we used OAuth2PasswordBearer.


