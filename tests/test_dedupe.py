from wellspring.dedupe import EntityResolver
from wellspring.storage.sqlite_store import SQLiteGraphStore


def test_entity_dedupe_by_key():
    store = SQLiteGraphStore(":memory:")
    resolver = EntityResolver(store)
    e1 = resolver.resolve("Ada Lovelace", "person")
    e2 = resolver.resolve("Ada Lovelace", "person")
    assert e1.id == e2.id
