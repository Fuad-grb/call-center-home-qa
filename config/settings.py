"""
Application configuration.
Loads settings from environment variables (.env file).
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All settings in one place. Values come from .env file."""

    # --- LLM ---
    groq_api_key: str = Field(default="", description="Groq API key")
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.1)  # Low = more deterministic
    llm_max_tokens: int = Field(default=1024)
    llm_max_retries: int = Field(default=3)

    # --- Pipeline ---
    min_segment_duration: float = Field(default=0.1)  # seconds

    # --- Logging ---
    log_level: str = Field(default="INFO")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this everywhere
settings = Settings()
