"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.endpoints import dictionary
from app.core.config import settings


# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(settings.DICTIONARY_SAVE_PATH), exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(dictionary.router, prefix=f"{settings.API_V1_STR}/dictionary", tags=["dictionary"])


@app.get("/", tags=["status"])
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {
        "status": "ok",
        "message": "Elastic Dictionary API is running",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 