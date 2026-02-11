from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    db_path: str = os.getenv("DB_PATH", "/data/wellspring.db")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_cooccurrence: bool = os.getenv("ENABLE_COOCCURRENCE", "0") == "1"
    cooccurrence_max_entities: int = int(os.getenv("CO_OCCURRENCE_MAX_ENTITIES", "25"))
    enable_inference: bool = os.getenv("ENABLE_INFERENCE", "0") == "1"


def get_settings() -> Settings:
    return Settings()
