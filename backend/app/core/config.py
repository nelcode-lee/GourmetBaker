"""
Application configuration using Pydantic settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Cranswick Technical Standards Agent"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # RAG Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 15  # Increased to get more candidates for better coverage
    MIN_RELEVANCE_SCORE: float = 0.2  # Lowered to 0.2 to catch more relevant chunks (was 0.3, still too strict)
    USE_QUERY_VARIATIONS: bool = True  # Enable by default for better query understanding
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Cache Settings
    CACHE_ENABLED: bool = True  # Enable query/response caching
    CACHE_TTL_SECONDS: int = 3600  # Cache TTL in seconds (default 1 hour)
    
    # Storage
    UPLOAD_DIR: str = "./uploads"
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

