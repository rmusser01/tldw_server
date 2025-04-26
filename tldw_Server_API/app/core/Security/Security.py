# /Server_API/app/core/Security.py
#
# Description: This file contains functions for hashing passwords and creating JWT tokens.
#
# Imports
from passlib.context import CryptContext
import time
#
# 3rd-Party Libraries
#from jose import JWTError, jwt
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

SECRET_KEY = "YOUR_SUPER_SECRET_KEY"  # load from config in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# FIXME - File is dummy code, needs to be updated

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# create_access_token adds an expiration (exp).
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


# decode_access_token returns the decoded payload if valid, or None if invalid/expired.
def decode_access_token(token: str):
    # JOSE library has a bug...
    # try:
    #     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    #     return payload
    # except JWTError:
    #     return None
    pass

#
# End of Security.py
# #####################################################################################################################