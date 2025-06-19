# /Server_API/app/schemas/auth.py


# FIXME - File is dummy code, needs to be updated

from pydantic import BaseModel, ConfigDict

class UserCreate(BaseModel):
    username: str
    email: str = None
    password: str

class UserRead(BaseModel):
    id: int
    username: str
    email: str = None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# UserCreate is what the client sends to register a new user.
#
# UserRead is what you return when reading a user from the DB (ID, username, etc.).
#
# Token is the structure for returning JWT tokens.