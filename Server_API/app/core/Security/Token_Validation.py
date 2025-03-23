
# FIXME - placeholder for actual logic
from fastapi import Depends, HTTPException, status
from jose import JWTError

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username: str = payload.get("sub")  # "sub" is subject
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    # fetch user from DB
    user_dict = get_user_by_username(username)
    if not user_dict:
        raise HTTPException(status_code=401, detail="User not found")

    # optionally check if is_active, etc.
    return user_dict



# If the token is invalid/expired, we raise a 401.
#
# Otherwise, we find the user in DB. Return that user as the “current user.”