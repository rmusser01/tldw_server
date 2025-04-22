# Server_API/app/api/v1/endpoints/auth.py
# Description: This code provides a FastAPI endpoint for user authentication using OAuth2 with password flow.
#
# Imports
from typing import Any
#
# # 3rd-party Libraries
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer

from tldw_Server_API.app.api.v1.schemas.auth import Token
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_current_user
from tldw_Server_API.app.core.DB_Management.Users_DB import get_user_by_username
from tldw_Server_API.app.core.Security.Security import verify_password, create_access_token

#
# Local Imports
#
#######################################################################################################################
#
# Functions:
router = APIRouter()


# FIXME - placeholder for actual logic


# /Server_API/app/api/v1/endpoints/auth.py
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


#
# # End of auth.py
#######################################################################################################################
