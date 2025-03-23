# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
#
# 3rd-party Libraries
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
#
# Local Imports
from .api.v1.endpoints.media import router as media_router
#
########################################################################################################################
#
# Functions:

app = FastAPI(
    title="tldw API",
    version="0.1.0",
    description="FastAPI Backend for the tldw project"
)

# -- If you have any global middleware, add it here --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the tldw API"}

# Then, include routers for your resources, for example:
app.include_router(media_router, prefix="/api/v1/media", tags=["media"])

