from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from .schemas import Subgraph, SubgraphEdge


def _normalize_cap(value: Optional[int]) -> Optional[int]:
    """Treat missing/zero/negative values as uncapped."""
    if value is None or value <= 0:
        return None
    return value


def _node_adjacency(edges: List[SubgraphEdge]) -> Dict[str, List[SubgraphEdge]]:
    adjacency: Dict[str, List[SubgraphEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.subject_id].append(edge)
        adjacency[edge.object_id].append(edge)
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda e: (-e.confidence, e.id))
    return adjacency


def _seeded_node_sample(
    subgraph: Subgraph,
    *,
    seed_id: str,
    node_cap: int,
) -> set[str]:
    """Pick up to node_cap nodes, preferring the seeded neighborhood first."""
    node_by_id = {node.id: node for node in subgraph.nodes}
    adjacency = _node_adjacency(subgraph.edges)

    selected: set[str] = set()
    queue: deque[str] = deque()
    if seed_id in node_by_id:
        queue.append(seed_id)

    while queue and len(selected) < node_cap:
        node_id = queue.popleft()
        if node_id in selected or node_id not in node_by_id:
            continue
        selected.add(node_id)
        if len(selected) >= node_cap:
            break
        for edge in adjacency.get(node_id, []):
            neighbor_id = (
                edge.object_id if edge.subject_id == node_id else edge.subject_id
            )
            if neighbor_id not in selected:
                queue.append(neighbor_id)

    if len(selected) < node_cap:
        for node in subgraph.nodes:
            if node.id in selected:
                continue
            selected.add(node.id)
            if len(selected) >= node_cap:
                break

    return selected


def limit_subgraph(
    subgraph: Subgraph,
    *,
    seed_id: str,
    max_nodes: Optional[int],
    max_edges: Optional[int],
) -> Tuple[Subgraph, bool]:
    """
    Bound a subgraph for UI rendering while preserving seed-local structure first.
    Returns (possibly-limited-subgraph, truncated_flag).
    """
    node_cap = _normalize_cap(max_nodes)
    edge_cap = _normalize_cap(max_edges)

    original_node_count = len(subgraph.nodes)
    original_edge_count = len(subgraph.edges)
    node_cap_triggered = node_cap is not None and original_node_count > node_cap
    edge_cap_triggered = edge_cap is not None and original_edge_count > edge_cap

    if not node_cap_triggered and not edge_cap_triggered:
        return subgraph, False

    if node_cap_triggered:
        selected_node_ids = _seeded_node_sample(
            subgraph,
            seed_id=seed_id,
            node_cap=node_cap or original_node_count,
        )
    else:
        selected_node_ids = {node.id for node in subgraph.nodes}

    selected_edges = [
        edge
        for edge in subgraph.edges
        if edge.subject_id in selected_node_ids and edge.object_id in selected_node_ids
    ]

    if edge_cap_triggered:
        selected_edges = sorted(
            selected_edges,
            key=lambda e: (-e.confidence, e.id),
        )[: edge_cap or len(selected_edges)]
        connected_node_ids: set[str] = set()
        for edge in selected_edges:
            connected_node_ids.add(edge.subject_id)
            connected_node_ids.add(edge.object_id)
        if seed_id:
            connected_node_ids.add(seed_id)
        selected_node_ids = selected_node_ids.intersection(connected_node_ids)

    limited_nodes = [node for node in subgraph.nodes if node.id in selected_node_ids]
    limited = Subgraph(nodes=limited_nodes, edges=selected_edges)
    truncated = (
        len(limited.nodes) < original_node_count
        or len(limited.edges) < original_edge_count
    )
    return limited, truncated
