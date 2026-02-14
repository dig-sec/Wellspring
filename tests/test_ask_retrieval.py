from __future__ import annotations

from dataclasses import dataclass

from mimir.api.ask_retrieval import extract_search_terms, gather_entities


@dataclass
class _Entity:
    id: str
    name: str
    type: str


def test_extract_search_terms_pulls_entity_candidate_from_question():
    terms = extract_search_terms("What can you tell me about dynowiper?")

    assert terms
    assert terms[0].lower() == "dynowiper"
    assert "dynowiper" in [term.lower() for term in terms]


def test_extract_search_terms_includes_structured_cti_tokens():
    terms = extract_search_terms("Any links between cve-2025-12345 and t1486?")
    lowered = [term.lower() for term in terms]

    assert "cve-2025-12345" in lowered
    assert "t1486" in lowered


def test_gather_entities_finds_and_dedupes_matches_across_terms():
    entity = _Entity(id="e-1", name="Dynowiper", type="malware")
    queries: list[str] = []

    def _search(query: str):
        queries.append(query)
        if query.lower() in {"dynowiper", "wiper"}:
            return [entity]
        return []

    found, terms = gather_entities(
        "What can you tell me about dynowiper wiper?",
        _search,
        limit=20,
    )

    assert found == [entity]
    assert "dynowiper" in [term.lower() for term in terms]
    assert len(queries) >= 1


def test_gather_entities_uses_punctuation_normalized_query_variants():
    entity = _Entity(id="e-2", name="Dynowiper", type="malware")
    queries: list[str] = []

    def _search(query: str):
        queries.append(query)
        if query.lower() == "dynowiper":
            return [entity]
        return []

    found, _terms = gather_entities(
        "What can you tell me about dyno-wiper?",
        _search,
        limit=20,
    )

    assert found == [entity]
    assert any(q.lower() == "dynowiper" for q in queries)
