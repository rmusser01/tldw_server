# /Server_API/app/api/v1/endpoints/media.py
from typing import Dict, List, Any

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