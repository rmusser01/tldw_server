# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
import logging
#
# 3rd-party Libraries
import sys
from pathlib import Path
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

#
# Local Imports
# Media Endpoint
from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
#
# RAG Endpoint
from tldw_Server_API.app.api.v1.endpoints.rag import router as rag_router
#
# Chat Endpoint
from tldw_Server_API.app.api.v1.endpoints.chat import router as chat_router
#
# Prompt Management Endpoint
from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#
########################################################################################################################
#
# Functions:


# --- Loguru Configuration with Intercept Handler ---

# Define a handler class to intercept standard logging messages
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# Remove default handler
logger.remove()

# Add your desired Loguru sink (e.g., stderr)
log_level = "INFO"
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# Configure standard logging to use the InterceptHandler
loggers_to_intercept = ["uvicorn", "uvicorn.error", "uvicorn.access"] # Add other library names if needed
for logger_name in loggers_to_intercept:
    mod_logger = logging.getLogger(logger_name)
    mod_logger.handlers = [InterceptHandler()]
    mod_logger.propagate = False # Prevent messages from reaching the root logger
    # Optionally set level if you only want certain levels from that lib
    # mod_logger.setLevel(logging.DEBUG)

logger.info("Loguru logger configured with SPECIFIC standard logging interception!")


BASE_DIR     = Path(__file__).resolve().parent
FAVICON_PATH = BASE_DIR / "static" / "favicon.ico"
app = FastAPI(
    title="tldw API",
    version="0.0.1",
    description="Version 0.0.1: Smooth Slide | FastAPI Backend for the tldw project"
)

# Static files serving
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Favicon serving
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")



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


# Router for RAG endpoints
app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])


# Router for chat endpoints/chat temp-file handling
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])


# Router for Prompt Management endpoints
app.include_router(prompt_router, prefix="/api/v1/prompts", tags=["prompts]"])


# Router for trash endpoints - deletion of media items / trash file handling (FIXME: Secure delete vs lag on delete?)
#app.include_router(trash_router, prefix="/api/v1/trash", tags=["trash"])

# Router for authentication endpoint
#app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
# The docs at http://localhost:8000/docs will show an “Authorize” button. You can log in by calling POST /api/v1/auth/login with a form that includes username and password. The docs interface is automatically aware because we used OAuth2PasswordBearer.


