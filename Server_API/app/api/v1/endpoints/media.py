# /Server_API/app/api/v1/endpoints/media.py
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Query, HTTPException
from Server_API.app.core.DB_Management.DB_Manager import add_media_to_database, search_media_db, \
    fetch_item_details_single

router = APIRouter()

# 1) Example Pydantic schema for creation
class MediaCreate(BaseModel):
    url: str
    info_dict: Dict[str, Any]  # e.g. {"title": "my video title", "uploader": "someone", ...}
    segments: List[Dict[str, Any]]  # e.g. a list of transcript segments
    summary: str
    keywords: str
    custom_prompt_input: str
    whisper_model: str
    overwrite: bool = False

# 2) Endpoint that calls the DB function
@router.post("/", summary="Create a new media record")
def create_media(payload: MediaCreate):
    """
    Calls add_media_to_database() from DB_Manager.py with the data from the request body.
    """
    try:
        result = add_media_to_database(
            url=payload.url,
            info_dict=payload.info_dict,
            segments=payload.segments,
            summary=payload.summary,
            keywords=payload.keywords,
            custom_prompt_input=payload.custom_prompt_input,
            whisper_model=payload.whisper_model,
            overwrite=payload.overwrite
        )
        return {"detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", summary="Get all media")
async def get_all_media():
    # For now, just return a placeholder list
    return [
        {"id": 1, "title": "Test Media 1"},
        {"id": 2, "title": "Another Media Item"},
    ]

# The MediaCreate schema dictates what the client must send in JSON form.
# We then unpack those fields and pass them to add_media_to_database() (which lives in DB_Manager.py).
# On success, we return {"detail": result}, or raise a 500 error if something fails.


@router.get("/", summary="List/search media")
def list_media(
    search_query: str = Query(None, description="Search term"),
    keywords: str = Query(None, description="Comma-separated keywords"),
    page: int = Query(1, ge=1, description="Page number"),
    results_per_page: int = Query(10, ge=1, description="Results per page")
):
    """
    Calls DB_Manager.search_media_db() with user-provided query parameters.
    """
    try:
        # your search_media_db function might want a list of fields or some other structure
        search_fields = ["title", "content"]  # or pass something else
        if not search_query and not keywords:
            # if no search query or keywords, you might want to just do a broad search
            # or handle it differently
            return {"detail": "No query params provided"}

        # Call your DB logic
        results = search_media_db(
            search_query=search_query or "",
            search_fields=search_fields,
            keywords=keywords or "",
            page=page,
            results_per_page=results_per_page
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# We define query parameters for search_query, keywords, page, etc.
# We pass them to DB_Manager.search_media_db().
# We return the raw results or transform them as needed.


@router.get("/{media_id}", summary="Get details about a single media item")
def get_media_item(media_id: int):
    """
    Calls DB_Manager.fetch_item_details_single() to get the media's prompt, summary, content, etc.
    """
    try:
        prompt, summary, content = fetch_item_details_single(media_id)
        return {
            "media_id": media_id,
            "prompt": prompt,
            "summary": summary,
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# 3.4. Replace “Standard” SQLAlchemy Patterns with Your DB Library
#
# In a typical FastAPI + SQLAlchemy setup, you’d inject a Session object via Depends(get_db). However:
#
#     You have a single db object that’s effectively a wrapper around your SQLite connection in DB_Manager.py.
#
#     The “CRUD” style is replaced by function calls like add_media_to_database(), check_media_exists(), search_media_db(), etc.
#
# So all you need is:
#
#     Import from DB_Manager.py.
#
#     Create a Pydantic schema for the request body (if needed).
#
#     Write an endpoint that calls your library function in a try/except.
#
#     Return the result (or raise HTTPException if an error occurs).
#
# That’s the entire integration step.



# Suppose your ingestion code is here:
from app.services.video_processing_service import process_video_task
# Or from App_Function_Libraries.Video_DL_Ingestion_Lib import parse_and_expand_urls,
class VideoIngestRequest(BaseModel):
    url: str
    whisper_model: Optional[str] = "medium"
    custom_prompt: Optional[str] = None
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    keywords: Optional[List[str]] = []
    diarize: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    include_timestamps: bool = True
    keep_original_video: bool = False

# FIXME - This is a dummy implementation. Replace with actual logic
@router.post("/process-video", summary="Process a video by URL")
async def process_video_endpoint(payload: VideoIngestRequest):
    """
    Ingests a video, runs transcription and summarization, stores results in DB.
    """
    try:
        result = await process_video_task(
            url=payload.url,
            whisper_model=payload.whisper_model,
            custom_prompt=payload.custom_prompt,
            api_name=payload.api_name,
            api_key=payload.api_key,
            keywords=payload.keywords,
            diarize=payload.diarize,
            start_time=payload.start_time,
            end_time=payload.end_time,
            include_timestamps=payload.include_timestamps,
            keep_original_video=payload.keep_original_video,
        )
        if not result:
            raise HTTPException(status_code=500, detail="Video processing failed or returned False")
        return {"detail": "Video processed successfully", "url": payload.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Per-user DB Handling
from fastapi import APIRouter, Depends, Header
from app.dependencies.database import get_user_db
from app.core.DB_Management.DB_Manager import add_media_to_database

router = APIRouter()


@router.post("/add")
def add_media(
        db_name: str,
        url: str,
        token: str = Header(..., description="Bearer token in Authorization header or pass it explicitly"),
):
    """
    Suppose we pass which DB we want in 'db_name' param,
    and read the token from the 'Authorization' header.
    """
    # 1) get the DB instance for the current user & the requested db_name
    user_db = get_user_db(token, db_name)

    # 2) now use user_db in place of your typical "db" object
    # or pass it to your DB_Manager functions if they accept a db instance
    # or if your DB_Manager uses a global approach, override the path, etc.

    # Example call
    result = add_media_to_database(
        url=url,
        info_dict={},
        segments=[],
        summary="",
        keywords="",
        custom_prompt_input="",
        whisper_model="",
        overwrite=False,
        db=user_db  # pass the instance
    )
    return {"detail": result}




# Ephemeral vs persistent media store
# /app/api/v1/endpoints/media.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.services.video_processing import process_video_file, process_video_url
from app.services.ephemeral_store import ephemeral_storage

router = APIRouter()

class MediaProcessUrlRequest(BaseModel):
    url: str
    mode: str = "persist"  # "ephemeral" or "persist"
    # plus any other optional fields like diarize, whisper_model, etc.
    diarize: bool = False
    whisper_model: str = "medium"
    # etc...


class MediaProcessUrlResponse(BaseModel):
    status: str
    media_id: Optional[str] = None
    # If ephemeral, might not store ID at all
    # or use a different ephemeral ID
    # plus maybe some metadata


@router.post("/process/url", response_model=MediaProcessUrlResponse)
async def process_media_url_endpoint(payload: MediaProcessUrlRequest, background_tasks: BackgroundTasks):
    """
    Ingests/transcribes a remote video by URL.
    Depending on 'mode', either store ephemeral or in DB.
    """
    try:
        # 1) Possibly spawn a background task if you want the call to return immediately
        #    For now, let's do synchronous to show the pattern.

        # result would contain transcript, metadata, etc.
        result = process_video_url(
            url=payload.url,
            diarize=payload.diarize,
            whisper_model=payload.whisper_model,
            # ... pass other fields
        )

        # 2) If ephemeral, store in ephemeral_storage. If persist, store in DB
        if payload.mode == "ephemeral":
            # store in ephemeral_store
            ephemeral_id = ephemeral_storage.store_data(result)
            return MediaProcessUrlResponse(
                status="ephemeral-ok",
                media_id=ephemeral_id
            )
        else:
            # store in DB, returning a numeric or string ID
            media_id = store_in_db(result)  # implement your DB logic
            return MediaProcessUrlResponse(
                status="persist-ok",
                media_id=str(media_id)  # convert to str for uniformity
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Handling File uploads vs URLs
# /app/api/v1/endpoints/media.py

class MediaProcessFileResponse(BaseModel):
    status: str
    media_id: Optional[str] = None

@router.post("/process/file", response_model=MediaProcessFileResponse)
async def process_media_file_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("persist"),   # ephemeral or persist
    diarize: bool = Form(False),
    whisper_model: str = Form("medium")
    # etc...
):
    """
    Ingest/transcribe an uploaded file.
    If 'mode=ephemeral', store ephemeral; otherwise DB.
    """
    try:
        # 1) Save temp file or read it in memory
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as out_f:
            content = await file.read()
            out_f.write(content)

        # 2) process file
        result = process_video_file(
            filepath=temp_path,
            diarize=diarize,
            whisper_model=whisper_model
            # ...
        )

        # 3) ephemeral vs persist
        if mode == "ephemeral":
            ephemeral_id = ephemeral_storage.store_data(result)
            return MediaProcessFileResponse(
                status="ephemeral-ok",
                media_id=ephemeral_id
            )
        else:
            media_id = store_in_db(result)
            return MediaProcessFileResponse(
                status="persist-ok",
                media_id=str(media_id)
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Media ingestion and analysis in a single endpoint
# /app/api/v1/endpoints/media.py (or a separate place)
class ProcessAndAnalyzeRequest(BaseModel):
    url: Optional[str] = None
    file: Optional[str] = None  # or handle differently
    mode: str = "persist"
    # Additional fields for transcription
    diarize: bool = False
    whisper_model: str = "medium"
    # Additional fields for analysis
    api_key: str = ""
    model: str = "gpt-4"

class ProcessAndAnalyzeResponse(BaseModel):
    media_id: Optional[str]
    summary: str

@router.post("/process-and-analyze", response_model=ProcessAndAnalyzeResponse)
def process_and_analyze(payload: ProcessAndAnalyzeRequest):
    """
    All in one ingestion -> transcription -> LLM summarization
    """
    # 1) If file is provided, handle that, else if url is provided, handle that
    #    (this is just a sample approach)
    if payload.url:
        processing_result = process_video_url(
            url=payload.url,
            diarize=payload.diarize,
            whisper_model=payload.whisper_model,
        )
    else:
        # handle file, or raise an error
        ...

    # 2) ephemeral vs. persist
    if payload.mode == "ephemeral":
        ephemeral_id = ephemeral_storage.store_data(processing_result)
        text_to_analyze = processing_result["transcript"]
        media_id = ephemeral_id
    else:
        # store in DB
        db_id = store_in_db(processing_result)
        text_to_analyze = processing_result["transcript"]
        media_id = str(db_id)

    # 3) Analysis
    summary = run_analysis(text_to_analyze, payload.api_key, payload.model)

    return ProcessAndAnalyzeResponse(
        media_id=media_id,
        summary=summary
    )




