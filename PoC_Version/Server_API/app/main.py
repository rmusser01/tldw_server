# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
#
# 3rd-party Libraries
from fastapi import FastAPI
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# Usage: uvicorn main:app --reload
app = FastAPI(title="tldw API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Welcome to the tldw API"}


