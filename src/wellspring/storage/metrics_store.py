from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MetricsStore(ABC):
    @abstractmethod
    def rollup_daily_threat_actor_stats(
        self,
        lookback_days: int = 365,
        min_confidence: float = 0.0,
        source_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_rollup_overview(
        self,
        days: int = 30,
        source_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError
