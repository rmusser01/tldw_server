







#OAuth2PasswordBearer Security Dependency
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# FIXME - placeholder for actual logic
# more OAuth2 stuff
# /Server_API/app/api/v1/endpoints/auth.py

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any

from app.core.security import verify_password, create_access_token
from app.core.DB_Management.DB_Manager import get_user_by_username
from ...schemas.auth import Token

router = APIRouter()

@router.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, returns JWT token.
    OAuth2PasswordRequestForm has fields: username, password, scope, client_id, client_secret
    """
    user = get_user_by_username(form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}





# OAuth2PasswordRequestForm automatically reads username and password from form data in application/x-www-form-urlencoded.
# We call verify_password(plain, hashed).
# On success, we create a token with {"sub": user["username"]} as the payload.
# We return the standard shape: {"access_token": "...", "token_type": "bearer"}.


#Any endpoint that needs a logged in user to access:
@router.get("/protected-route")
def read_protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {current_user['username']}!"}

# If a client calls this endpoint without a valid “Authorization: Bearer <token>”, they’ll get a 401 error.
# If the token is valid, current_user is populated with the user record from DB.



