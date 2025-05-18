from sqlalchemy import Column, Integer, String, Boolean, DateTime
from Server_API.db.database import Base

class Media(Base):
    __tablename__ = "media"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    url = Column(String)
    content = Column(String)
    is_trash = Column(Boolean, default=False)
    trash_date = Column(DateTime, nullable=True)