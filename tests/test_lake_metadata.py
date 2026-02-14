from __future__ import annotations

from datetime import datetime, timezone

from mimir.lake import build_lake_metadata, infer_source, parse_source_uri


def test_infer_source_uses_uri_scheme():
    assert infer_source("opencti://report/abc") == "opencti"
    assert infer_source("elasticsearch://idx/doc-1") == "elasticsearch"
    assert infer_source("upload://report.txt") == "upload"
    assert infer_source("") == "unknown"


def test_parse_source_uri_extracts_collection_and_record():
    parsed = parse_source_uri("opencti://report/opencti--report--123")
    assert parsed["source"] == "opencti"
    assert parsed["collection"] == "report"
    assert parsed["record_id"] == "opencti--report--123"

    parsed = parse_source_uri("elasticsearch://feedly_news/doc-42")
    assert parsed["source"] == "elasticsearch"
    assert parsed["collection"] == "feedly_news"
    assert parsed["record_id"] == "doc-42"

    parsed = parse_source_uri("file:///tmp/a.txt")
    assert parsed["source"] == "file"
    assert parsed["collection"] == "filesystem"
    assert parsed["record_id"] == "/tmp/a.txt"


def test_build_lake_metadata_enriches_existing_metadata_without_dropping_fields():
    started = datetime(2026, 2, 14, 12, 30, 0, tzinfo=timezone.utc)
    metadata = {
        "title": "Example",
        "connector": "opencti",
    }
    enriched = build_lake_metadata(
        "opencti://report/opencti--report--123",
        metadata,
        ingested_at=started,
    )

    assert enriched["title"] == "Example"
    assert enriched["connector"] == "opencti"
    assert enriched["source"] == "opencti"
    assert enriched["lake"]["source"] == "opencti"
    assert enriched["lake"]["collection"] == "report"
    assert enriched["lake"]["record_id"] == "opencti--report--123"
    assert enriched["lake"]["source_uri"] == "opencti://report/opencti--report--123"
    assert enriched["lake"]["ingested_at"] == "2026-02-14T12:30:00+00:00"
    assert enriched["lake"]["version"] == 1
