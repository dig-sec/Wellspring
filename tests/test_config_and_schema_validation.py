from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from mimir.config import get_settings, validate_settings
from mimir.schemas import QueryRequest


def test_validate_settings_rejects_short_search_query_max_length():
    settings = replace(get_settings(), search_query_max_length=7)
    with pytest.raises(ValueError, match="SEARCH_QUERY_MAX_LENGTH"):
        validate_settings(settings)


def test_query_request_depth_bounds():
    QueryRequest(depth=0)
    QueryRequest(depth=5)

    with pytest.raises(ValidationError):
        QueryRequest(depth=-1)

    with pytest.raises(ValidationError):
        QueryRequest(depth=6)
