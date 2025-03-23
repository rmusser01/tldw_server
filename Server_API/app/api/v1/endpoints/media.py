# /Server_API/app/api/v1/endpoints/media.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/", summary="Get all media")
async def get_all_media():
    # For now, just return a placeholder list
    return [
        {"id": 1, "title": "Test Media 1"},
        {"id": 2, "title": "Another Media Item"},
    ]
