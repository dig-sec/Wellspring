from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "phi4")
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    elastic_hosts: str = os.getenv("ELASTICSEARCH_HOST", "http://127.0.0.1:9200")
    elastic_user: str = os.getenv("ELASTICSEARCH_USER", "")
    elastic_password: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    elastic_index_prefix: str = os.getenv("ELASTICSEARCH_INDEX_PREFIX", "wellspring")
    elastic_verify_certs: bool = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "1") == "1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_cooccurrence: bool = os.getenv("ENABLE_COOCCURRENCE", "0") == "1"
    cooccurrence_max_entities: int = int(os.getenv("CO_OCCURRENCE_MAX_ENTITIES", "25"))
    enable_inference: bool = os.getenv("ENABLE_INFERENCE", "0") == "1"
    max_chunks_per_run: int = int(os.getenv("MAX_CHUNKS_PER_RUN", "50"))
    metrics_rollup_enabled: bool = os.getenv("METRICS_ROLLUP_ENABLED", "1") == "1"
    metrics_rollup_interval_seconds: int = int(os.getenv("METRICS_ROLLUP_INTERVAL_SECONDS", "900"))
    metrics_rollup_lookback_days: int = int(os.getenv("METRICS_ROLLUP_LOOKBACK_DAYS", "365"))
    metrics_rollup_min_confidence: float = float(os.getenv("METRICS_ROLLUP_MIN_CONFIDENCE", "0.0"))
    opencti_url: str = os.getenv("OPENCTI_URL", "")
    opencti_token: str = os.getenv("OPENCTI_TOKEN", "")
    watched_folders: str = os.getenv("WATCHED_FOLDERS", "/data/documents")

    @property
    def elastic_hosts_list(self) -> List[str]:
        return [host.strip() for host in self.elastic_hosts.split(",") if host.strip()]

    @property
    def watched_folders_list(self) -> List[str]:
        return [d.strip() for d in self.watched_folders.split(",") if d.strip()]


def get_settings() -> Settings:
    return Settings()
