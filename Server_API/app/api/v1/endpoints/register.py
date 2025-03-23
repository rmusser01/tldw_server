# /Server_API/app/api/v1/endpoints/auth.py

#FIXME - File is dummy code, needs to be updated


from app.core.security import hash_password
from app.core.DB_Management.DB_Manager import create_user
from ...schemas.auth import UserCreate, UserRead

@router.post("/register", response_model=UserRead)
def register_user(user_in: UserCreate):
    existing = get_user_by_username(user_in.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = hash_password(user_in.password)
    user_id = create_user(user_in.username, hashed, user_in.email)

    return {
        "id": user_id,
        "username": user_in.username,
        "email": user_in.email,
        "is_active": True
    }
