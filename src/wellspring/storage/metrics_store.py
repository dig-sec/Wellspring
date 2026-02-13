from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
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

    @abstractmethod
    def rollup_daily_pir_stats(
        self,
        lookback_days: int = 365,
        min_confidence: float = 0.0,
        source_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_pir_trending_summary(
        self,
        *,
        days: int = 7,
        top_n: int = 10,
        source_uri: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError
