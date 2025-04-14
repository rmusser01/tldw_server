# Media_Update_lib.py
# Description: File contains functions relating to updating media items in the database.
#
# Imports
import sqlite3
from typing import Optional, List
#
# 3rd-party Libraries
from fastapi import HTTPException, Depends
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.DB_Dependency import get_db_manager
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_full_media_details2, create_document_version, \
    update_keywords_for_media
#
########################################################################################################################
#
# Functions:

def process_media_update(
    media_id: int,
    content: Optional[str] = None,
    prompt: Optional[str] = None,
    summary: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    db=Depends(get_db_manager)
):
    """Centralized media update processing"""
    try:
        # Verify media exists
        existing = get_full_media_details2(media_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Media not found")

        # Process content updates
        if content is not None:
            create_document_version(
                media_id=media_id,
                content=content,
                prompt=prompt or existing.get('prompt'),
                summary=summary or existing.get('summary'),
                db=db
            )

        # Process metadata updates
        updates = {}
        if prompt is not None:
            updates['prompt'] = prompt
        if summary is not None:
            updates['summary'] = summary

        if updates:
            with db.transaction() as conn:
                cursor = conn.cursor()
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                cursor.execute(
                    f"UPDATE Media SET {set_clause} WHERE id = ?",
                    list(updates.values()) + [media_id]
                )

        # Process keyword updates
        if keywords is not None:
            update_keywords_for_media(media_id, keywords, db=db)

        return get_full_media_details2(media_id)

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

#
# End of Media_Update_lib.py
########################################################################################################################
