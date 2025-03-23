





# FIXME - This is a dummy implementation. Replace with actual logic
def create_user(username: str, hashed_password: str, email: str = None) -> int:
    """
    Inserts a new user into the Users table. Returns the new user ID.
    """
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Users (username, hashed_password, email) VALUES (?, ?, ?)",
            (username, hashed_password, email)
        )
        conn.commit()
        return cursor.lastrowid

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, hashed_password, email, is_active FROM Users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "username": row[1],
                "hashed_password": row[2],
                "email": row[3],
                "is_active": bool(row[4])
            }
        return None



# Per-user DB code
# /Server_API/app/dependencies/database.py
from fastapi import Depends, HTTPException
from app.core.security import decode_access_token
from app.core.DB_Management.DB_Manager import Database  # or your custom class


def get_user_db_path(user_id: int, db_name: str) -> str:
    """
    Construct the file path for the user's chosen database.
    E.g.: db/user_{user_id}/{db_name}.db
    """
    return f"./db/user_{user_id}/{db_name}.db"


def get_user_db(
        token: str,
        db_name: str,
) -> Database:
    """
    1) Decode the token to find user_id
    2) Construct a path for the user's DB
    3) Return a Database instance pointing to that file
    """
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # 'sub' is typically the username, or you could store the user_id in the token’s payload
    # For a more robust approach, fetch user from DB, get user_id from there, etc.
    user_id = payload.get("user_id")  # or fetch it from your user table by username

    path = get_user_db_path(user_id=user_id, db_name=db_name)
    db_instance = Database(os.path.basename(path))  # or pass the entire path
    return db_instance
