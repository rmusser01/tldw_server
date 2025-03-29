# /Server_API/app/api/v1/endpoints/trash.py
from fastapi import APIRouter, HTTPException

from app.core.DB_Management.DB_Manager import mark_as_trash, get_trashed_items, restore_from_trash, permanently_delete_item

router = APIRouter()

@router.get("/", summary="List trashed items")
def list_trashed_items():
    return get_trashed_items()  # returns a list of dict

@router.post("/{media_id}/restore", summary="Restore an item from trash")
def restore_item(media_id: int):
    try:
        restore_from_trash(media_id)
        return {"detail": f"Media {media_id} restored from trash"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{media_id}", summary="Permanently delete trashed item")
def delete_item(media_id: int):
    try:
        permanently_delete_item(media_id)
        return {"detail": f"Media {media_id} permanently deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{media_id}/trash", summary="Move item to trash")
def trash_item(media_id: int):
    try:
        mark_as_trash(media_id)
        return {"detail": f"Media {media_id} moved to trash"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
