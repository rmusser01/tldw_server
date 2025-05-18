import os

def get_settings():
    return {
        "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///./tldw.db"),
        # Add other configuration variables as needed
    }

settings = get_settings()