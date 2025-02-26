"""
Configuration settings for the Elastic Dictionary application
"""
from pydantic_settings import BaseSettings
import os
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Elastic Dictionary API"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Dictionary settings
    DICTIONARY_MODEL: str = "all-MiniLM-L6-v2"
    DICTIONARY_SAVE_PATH: str = "data/elastic_dict.pkl"
    
    # Ensure data directory exists
    @property
    def ensure_data_dir(self) -> None:
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.DICTIONARY_SAVE_PATH), exist_ok=True)
    
    class Config:
        case_sensitive = True


# Create global settings object
settings = Settings()
settings.ensure_data_dir 