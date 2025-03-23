# /Server_API/app/schemas/media.py
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime

from App_Function_Libraries.DB.DB_Manager import fetch_item_details_single
from Server_API.app.api.v1.endpoints.media import router


class MediaBase(BaseModel):
    title: str
    url: str
    content: Optional[str] = None
    is_trash: Optional[bool] = False
    trash_date: Optional[datetime] = None

class MediaCreate(MediaBase):
    """Schema for creating new media items"""
    pass

class MediaUpdate(MediaBase):
    """Schema for updating media items (could be partial)"""
    # If partial updates are needed, you can make all fields optional or handle patch-like logic
    pass

class MediaRead(MediaBase):
    id: int

    class Config:
        orm_mode = True



# MediaBase includes fields shared by creation, reading, and updating.
#
# MediaCreate extends MediaBase but doesn’t add anything new yet. (You might add required fields or validations that apply only to creation.)
#
# MediaUpdate extends MediaBase for partial updates. Often we set fields as optional in the update schema, but that’s up to your preference.
#
# MediaRead is what we return when reading from the database. Notice orm_mode = True so we can directly return SQLAlchemy objects.



class MediaRead(BaseModel):
    media_id: int
    prompt: str
    summary: str
    content: str

@router.get("/{media_id}", response_model=MediaRead)
def get_media_item(media_id: int):
    try:
        prompt, summary, content = fetch_item_details_single(media_id)
        return MediaRead(
            media_id=media_id,
            prompt=prompt,
            summary=summary,
            content=content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


