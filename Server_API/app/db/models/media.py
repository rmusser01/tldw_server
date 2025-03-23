# /Server_API/app/db/models/media.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from ..database import Base  # relative import from db.database

class Media(Base):
    __tablename__ = "media"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    url = Column(String)
    content = Column(String)
    is_trash = Column(Boolean, default=False)
    trash_date = Column(DateTime, nullable=True)
