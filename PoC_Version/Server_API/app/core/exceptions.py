from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class VideoProcessingError(Exception):
    pass

async def video_processing_exception_handler(request: Request, exc: VideoProcessingError):
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred during video processing: {str(exc)}"},
    )

def setup_exception_handlers(app):
    app.add_exception_handler(VideoProcessingError, video_processing_exception_handler)