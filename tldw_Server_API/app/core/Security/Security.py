# /Server_API/app/core/Security.py
#
# Description: This file contains functions for hashing passwords and creating/validating JWT tokens.
#
# Imports
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

# 3rd-Party Libraries
from passlib.context import CryptContext
import jwt # Using PyJWT library (pip install pyjwt)
from pydantic import BaseModel, ValidationError

# Local Imports
from tldw_Server_API.app.core.Utils.Utils import logging # Assuming your logging setup is here

#######################################################################################################################

# --- Configuration ---

# Load SECRET_KEY from environment variable. Fallback only for development/testing.
# WARNING: Never use the default key in production! Generate a strong, unique key.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_insecure_default_secret_key_for_dev_only")
ALGORITHM = "HS256" # Algorithm for signing JWTs
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Default token expiration time

# Check if the default secret key is being used and log a warning (important for production awareness)
if SECRET_KEY == "a_very_insecure_default_secret_key_for_dev_only":
    logging.warning("!!! SECURITY WARNING: Using default JWT_SECRET_KEY. Set a strong JWT_SECRET_KEY environment variable for production! !!! Unrelated to single-user mode.")

# --- Password Hashing ---
# CryptContext for password hashing using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str) -> str:
    """Hashes a plain text password using bcrypt."""
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

# --- JWT Handling ---

# Pydantic model for data extracted from the token payload
class TokenData(BaseModel):
    user_id: Optional[int] = None # Store user ID directly in the token

def create_access_token(data: dict, expires_delta_minutes: Optional[int] = None) -> str:
    """
    Creates a JWT access token.

    Args:
        data (dict): Data to encode in the token. MUST contain 'user_id'.
        expires_delta_minutes (Optional[int]): Custom expiration time in minutes.
                                                 Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        str: The encoded JWT access token.

    Raises:
        ValueError: If 'user_id' is missing in the input data.
    """
    to_encode = data.copy()
    if "user_id" not in to_encode:
        # Ensure the essential identifier is present before creating the token
        logging.error("Attempted to create token without 'user_id' in data.")
        raise ValueError("Input data for token creation must contain 'user_id'.")

    # Determine expiration time
    if expires_delta_minutes is not None:
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_delta_minutes)
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # Build payload: use 'sub' (subject) for user_id, add 'exp' (expiration) and 'iat' (issued at)
    payload = {
        "sub": str(to_encode["user_id"]), # Subject claim should be the user identifier (as string for JWT standard)
        "exp": expire,
        "iat": datetime.now(timezone.utc)
        # You can add other claims here if needed (e.g., roles, permissions)
        # "roles": ["user"]
    }
    # Note: Don't put the raw 'data' dict directly into the payload unless necessary.
    # Keep the payload lean with standard claims and essential custom claims.

    logging.debug(f"Creating token for user_id: {to_encode['user_id']} expiring at {expire}")
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decodes and validates a JWT access token.

    Args:
        token (str): The JWT token string.

    Returns:
        Optional[TokenData]: A Pydantic model containing the extracted user_id if the
                             token is valid and contains the 'sub' claim, otherwise None.
    """
    try:
        # Decode the token. PyJWT handles expiration ('exp') check automatically.
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM] # Specify the expected algorithm
        )

        # Extract the subject ('sub' claim), which we defined as the user_id
        user_id_str = payload.get("sub")

        if user_id_str is None:
             logging.warning("Token decoded successfully, but 'sub' (user_id) claim is missing.")
             return None

        # Convert user_id back to integer
        try:
             user_id_int = int(user_id_str)
             token_data = TokenData(user_id=user_id_int)
             logging.debug(f"Token successfully decoded for user_id: {token_data.user_id}")
             return token_data
        except (ValueError, TypeError):
             logging.warning(f"Token 'sub' claim '{user_id_str}' could not be converted to integer.")
             return None

    except jwt.ExpiredSignatureError:
        logging.warning("Token validation failed: Signature has expired.")
        return None
    except jwt.InvalidSignatureError:
        logging.error("Token validation failed: Invalid signature.") # More severe - potential tampering
        return None
    except jwt.InvalidAlgorithmError:
        logging.error(f"Token validation failed: Invalid algorithm. Expected {ALGORITHM}.")
        return None
    except jwt.InvalidTokenError as e: # Catch other potential JWT errors
        logging.warning(f"Token validation failed: Invalid token - {e}")
        return None
    except Exception as e: # Catch unexpected errors during decoding
        logging.error(f"Unexpected error during token decoding: {e}", exc_info=True)
        return None

#
# End of Security.py
# #####################################################################################################################
