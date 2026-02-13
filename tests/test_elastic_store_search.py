from __future__ import annotations

from mimir.storage.elastic_store import ElasticGraphStore, _ElasticIndices


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"hits": {"hits": []}}


def _make_store(client: _FakeClient) -> ElasticGraphStore:
    store = object.__new__(ElasticGraphStore)
    store.client = client
    store.indices = _ElasticIndices("test")
    return store


def test_search_entities_uses_safe_prefix_wildcard_query():
    client = _FakeClient()
    store = _make_store(client)

    result = store.search_entities("  ap?t*  ", entity_type="malware")

    assert result == []
    assert len(client.calls) == 1
    request = client.calls[0]
    bool_query = request["query"]["bool"]
    wildcard_value = bool_query["should"][1]["wildcard"]["name.keyword"]["value"]

    assert request["size"] == 50
    assert bool_query["minimum_should_match"] == 1
    assert bool_query["filter"] == [{"term": {"type": "malware"}}]
    assert wildcard_value.endswith("*")
    assert not wildcard_value.startswith("*")
    assert "\\?" in wildcard_value
    assert "\\*" in wildcard_value


def test_search_entities_uses_canonical_key_filter_path():
    client = _FakeClient()
    store = _make_store(client)

    result = store.search_entities(
        "",
        canonical_key="ada lovelace|person",
        entity_type="person",
    )

    assert result == []
    assert len(client.calls) == 1
    assert client.calls[0]["query"] == {
        "bool": {
            "filter": [
                {"term": {"keys": "ada lovelace|person"}},
                {"term": {"type": "person"}},
            ]
        }
    }
    assert client.calls[0]["size"] == 50


def test_search_entities_skips_blank_query():
    client = _FakeClient()
    store = _make_store(client)

    result = store.search_entities("   ")

    assert result == []
    assert client.calls == []
