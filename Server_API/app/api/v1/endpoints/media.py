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
    media_type: str = "video"   # can be "audio" or "video"
    mode: str = "persist"       # "ephemeral" or "persist"
    whisper_model: str = "medium"
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    keywords: Optional[List[str]] = []
    diarize: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    include_timestamps: bool = True
    keep_original: bool = False
    # Add any other fields relevant to audio or video


class MediaProcessUrlResponse(BaseModel):
    status: str
    media_id: Optional[str] = None
    # If ephemeral, might not store ID at all
    # or use a different ephemeral ID
    # plus maybe some metadata


@router.post("/process/url", summary="Process a remote media file by URL")
async def process_media_url_endpoint(
    payload: MediaProcessUrlRequest
):
    """
    Ingest/transcribe a remote file. Depending on 'media_type', use audio or video logic.
    Depending on 'mode', store ephemeral or in DB.
    """
    try:
        # Step 1) Distinguish audio vs. video ingestion
        if payload.media_type.lower() == "audio":
            # Call your audio ingestion logic
            result = process_audio_url(
                url=payload.url,
                whisper_model=payload.whisper_model,
                api_name=payload.api_name,
                api_key=payload.api_key,
                keywords=payload.keywords,
                diarize=payload.diarize,
                include_timestamps=payload.include_timestamps,
                keep_original=payload.keep_original,
                start_time=payload.start_time,
                end_time=payload.end_time,
            )
        else:
            # Default to video
            result = process_video_url(
                url=payload.url,
                whisper_model=payload.whisper_model,
                api_name=payload.api_name,
                api_key=payload.api_key,
                keywords=payload.keywords,
                diarize=payload.diarize,
                include_timestamps=payload.include_timestamps,
                keep_original_video=payload.keep_original,
                start_time=payload.start_time,
                end_time=payload.end_time,
            )

        # result is presumably a dict containing transcript, some metadata, etc.
        if not result:
            raise HTTPException(status_code=500, detail="Processing failed or returned no result")

        # Step 2) ephemeral vs. persist
        if payload.mode == "ephemeral":
            ephemeral_id = ephemeral_storage.store_data(result)
            return {
                "status": "ephemeral-ok",
                "media_id": ephemeral_id,
                "media_type": payload.media_type
            }
        else:
            # If you want to store in your main DB, do so:
            media_id = store_in_db(result)  # or add_media_to_database(...) from DB_Manager
            return {
                "status": "persist-ok",
                "media_id": str(media_id),
                "media_type": payload.media_type
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# The function process_audio_url(...) is not part of your code yet, but it’s easy to adapt from your existing download_audio_file(...) + transcription approach. You can unify that into a new function (like your process_audio_files(...), but specialized for a single URL).
# Similarly, store_in_db(...) might just call your add_media_with_keywords(...) or add_media_to_database(...).




# Handling File uploads vs URLs
# /app/api/v1/endpoints/media.py

# We can’t use BaseModel directly for `File` inputs, so we read them with `Form(...)`
# or keep them in separate function parameters.
# For ephemeral/persistent, chunking, etc., we can still do a "Form" field for each.

class MediaProcessFileResponse(BaseModel):
    status: str
    media_id: Optional[str] = None

@router.post("/process/file", response_model=MediaProcessFileResponse)
async def process_media_file_endpoint(
    file: UploadFile = File(...),
    media_type: str = Form("video"),  # or "audio"
    mode: str = Form("persist"),      # ephemeral or persist
    diarize: bool = Form(False),
    whisper_model: str = Form("medium"),
    api_name: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    keep_original: bool = Form(False),
    include_timestamps: bool = Form(True)
    # ... etc.
):
    """
    Ingest/transcribe an uploaded file.
    If media_type=audio, use audio logic; if video, use video logic.
    If 'mode=ephemeral', store ephemeral; otherwise store in DB.
    """
    try:
        # 1) Save the uploaded file to a temp location
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as out_f:
            content = await file.read()
            out_f.write(content)

        # 2) Decide audio vs. video
        if media_type.lower() == "audio":
            result = process_audio_file(
                filepath=tmp_path,
                whisper_model=whisper_model,
                api_name=api_name,
                api_key=api_key,
                diarize=diarize,
                include_timestamps=include_timestamps,
                keep_original=keep_original,
            )
        else:
            # default to video
            result = process_video_file(
                filepath=tmp_path,
                whisper_model=whisper_model,
                api_name=api_name,
                api_key=api_key,
                diarize=diarize,
                include_timestamps=include_timestamps,
                keep_original_video=keep_original
            )

        if not result:
            raise HTTPException(status_code=500, detail="Processing file failed or returned no result")

        # 3) ephemeral vs. persist
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

# Where do we get process_audio_file from?
#     You can create a helper in your “audio ingestion library” that runs “download → reencode → transcribe → (optionally) summarize → returns a dictionary with transcript + summary.”
# Where do we store the final results?
#     If ephemeral, we do ephemeral_storage.store_data(...).
#     If persistent, we call your DB manager code.


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




# Podcast Integration
# /Server_API/app/api/v1/endpoints/media.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

from app.services.podcast_processing_service import process_podcast_task
from app.services.ephemeral_store import ephemeral_storage
from app.core.DB_Management.DB_Manager import add_media_to_database

router = APIRouter()

class PodcastIngestRequest(BaseModel):
    url: str
    custom_prompt: Optional[str] = None
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    keywords: Optional[List[str]] = []
    diarize: bool = False
    whisper_model: str = "medium"
    keep_original_audio: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    include_timestamps: bool = True
    mode: str = "persist"  # or "ephemeral"
    cookies: Optional[str] = None  # for advanced usage if needed

@router.post("/process-podcast", summary="Ingest & process a podcast by URL")
async def process_podcast_endpoint(payload: PodcastIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingests a podcast from the given URL, transcribes and summarizes it.
    Depending on 'mode' (ephemeral or persist), store in ephemeral_store or DB.
    """
    try:
        # (1) Run the “podcast processing service”
        result_data = await process_podcast_task(
            url=payload.url,
            custom_prompt=payload.custom_prompt,
            api_name=payload.api_name,
            api_key=payload.api_key,
            keywords=payload.keywords,
            diarize=payload.diarize,
            whisper_model=payload.whisper_model,
            keep_original_audio=payload.keep_original_audio,
            start_time=payload.start_time,
            end_time=payload.end_time,
            include_timestamps=payload.include_timestamps,
            cookies=payload.cookies,
        )

        # (2) Ephemeral vs. persist
        if payload.mode == "ephemeral":
            ephemeral_id = ephemeral_storage.store_data(result_data)
            return {
                "status": "ephemeral-ok",
                "media_id": ephemeral_id,
                "podcast_title": result_data.get("podcast_title")
            }
        else:
            # store in DB
            # Build up the DB data structure
            info_dict = {
                "title": result_data.get("podcast_title"),
                "author": result_data.get("podcast_author"),
            }
            transcript_segments = [{"Text": seg["Text"]} for seg in result_data["segments"]]

            # Insert into DB
            media_id = add_media_to_database(
                url=payload.url,
                info_dict=info_dict,
                segments=transcript_segments,
                summary=result_data["summary"],
                keywords=",".join(payload.keywords),
                custom_prompt_input=payload.custom_prompt or "",
                whisper_model=payload.whisper_model,
                overwrite=False
            )
            return {
                "status": "persist-ok",
                "media_id": str(media_id),
                "podcast_title": result_data.get("podcast_title")
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Ebook Ingestion Endpoint
# /Server_API/app/api/v1/endpoints/media.py

from fastapi import UploadFile, File, Form
from app.services.ebook_processing_service import process_ebook_task

class EbookIngestRequest(BaseModel):
    # If you want to handle only file uploads, skip URL
    # or if you do both, you can add a union
    title: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[List[str]] = []
    custom_prompt: Optional[str] = None
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    mode: str = "persist"
    chunk_size: int = 500
    chunk_overlap: int = 200

@router.post("/process-ebook", summary="Ingest & process an e-book file")
async def process_ebook_endpoint(
    background_tasks: BackgroundTasks,
    payload: EbookIngestRequest = Form(...),
    file: UploadFile = File(...)
):
    """
    Ingests an eBook (e.g. .epub).
    You can pass all your custom fields in the form data plus the file itself.
    """
    try:
        # 1) Save file to a tmp path
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2) Process eBook
        result_data = await process_ebook_task(
            file_path=tmp_path,
            title=payload.title,
            author=payload.author,
            keywords=payload.keywords,
            custom_prompt=payload.custom_prompt,
            api_name=payload.api_name,
            api_key=payload.api_key,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap
        )

        # 3) ephemeral vs. persist
        if payload.mode == "ephemeral":
            ephemeral_id = ephemeral_storage.store_data(result_data)
            return {
                "status": "ephemeral-ok",
                "media_id": ephemeral_id,
                "ebook_title": result_data.get("ebook_title")
            }
        else:
            # If persisting to DB:
            info_dict = {
                "title": result_data.get("ebook_title"),
                "author": result_data.get("ebook_author"),
            }
            # Possibly you chunk the text into segments—just an example:
            segments = [{"Text": result_data["text"]}]
            media_id = add_media_to_database(
                url=file.filename,
                info_dict=info_dict,
                segments=segments,
                summary=result_data["summary"],
                keywords=",".join(payload.keywords),
                custom_prompt_input=payload.custom_prompt or "",
                whisper_model="ebook-imported",  # or something else
                overwrite=False
            )
            return {
                "status": "persist-ok",
                "media_id": str(media_id),
                "ebook_title": result_data.get("ebook_title")
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# If you want to also accept an eBook by URL (for direct download) instead of a file-upload, you can adapt the request model and your service function accordingly.
# If you want chunk-by-chapter logic or more advanced processing, integrate your existing import_epub(...) from Book_Ingestion_Lib.py.

