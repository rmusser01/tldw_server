# /Server_API/app/core/security.py (or anywhere else)

# FIXME - File is dummy code, needs to be updated

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# When creating a new user, do hashed = hash_password(plaintext_password) and store hashed.
#
# When authenticating, do verify_password(plaintext, user["hashed_password"]).


# /Server_API/app/core/security.py

import time
# FIXME - Bug in jose module
#from jose import JWTError, jwt

SECRET_KEY = "YOUR_SUPER_SECRET_KEY"  # load from config in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: int = None):
    # FIXME JOSE library has a bug
    # to_encode = data.copy()
    # if expires_delta:
    #     expire = int(time.time()) + expires_delta
    # else:
    #     expire = int(time.time()) + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    # to_encode.update({"exp": expire})
    # encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # return encoded_jwt
    pass

def decode_access_token(token: str):
    # JOSE library has a bug...
    # try:
    #     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    #     return payload
    # except JWTError:
    #     return None
    pass

# create_access_token adds an expiration (exp).
#
# decode_access_token returns the decoded payload if valid, or None if invalid/expired.