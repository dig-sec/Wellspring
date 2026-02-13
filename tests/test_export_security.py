from __future__ import annotations

import csv
import io
import zipfile

from mimir.api.visualize import render_html
from mimir.export import export_csv_zip
from mimir.schemas import Subgraph, SubgraphEdge, SubgraphNode


def _security_subgraph() -> Subgraph:
    nodes = [
        SubgraphNode(id="=source", name="=SUM(1,1)", type="@actor"),
        SubgraphNode(id="target", name="Target", type="tool"),
    ]
    edges = [
        SubgraphEdge(
            id="+edge",
            subject_id="=source",
            predicate="-danger",
            object_id="target",
            confidence=0.95,
            attrs={},
        )
    ]
    return Subgraph(nodes=nodes, edges=edges)


def test_export_csv_zip_neutralizes_formula_cells():
    payload = export_csv_zip(_security_subgraph())
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        entities_csv = archive.read("entities.csv").decode("utf-8")
        relations_csv = archive.read("relations.csv").decode("utf-8")

    entity_rows = list(csv.reader(io.StringIO(entities_csv)))
    relation_rows = list(csv.reader(io.StringIO(relations_csv)))

    assert entity_rows[1][0] == "'=source"
    assert entity_rows[1][1] == "'=SUM(1,1)"
    assert entity_rows[1][2] == "'@actor"
    assert relation_rows[1][0] == "'+edge"
    assert relation_rows[1][3] == "'-danger"


def test_render_html_escapes_title_and_script_breakout_sequences():
    subgraph = Subgraph(
        nodes=[
            SubgraphNode(
                id="n1",
                name="</script><script>alert(1)</script>",
                type="indicator",
            )
        ],
        edges=[],
    )
    html = render_html(subgraph, title='<img src=x onerror="alert(1)">')

    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "<img src=x onerror" not in html
    assert "</script><script>alert(1)</script>" not in html
    assert "<\\/script>" in html
