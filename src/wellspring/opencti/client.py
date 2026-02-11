"""OpenCTI GraphQL client for Wellspring integration."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class OpenCTIClient:
    """Synchronous GraphQL client for OpenCTI API.

    Designed to run in a background thread so it never blocks the
    async event loop.
    """

    def __init__(self, base_url: str, api_token: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "Wellspring/1.0",
                "Accept": "application/json",
            },
            timeout=timeout,
            verify=True,
        )

    def close(self):
        self._client.close()

    def query(self, gql: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query against OpenCTI."""
        payload = {"query": gql}
        if variables:
            payload["variables"] = variables

        resp = self._client.post(f"{self.base_url}/graphql", json=payload)

        if resp.status_code == 403:
            logger.error("OpenCTI 403: headers=%s body=%s", resp.headers, resp.text[:500])

        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            errors = data["errors"]
            logger.error("OpenCTI GraphQL errors: %s", errors)
            raise RuntimeError(f"OpenCTI GraphQL error: {errors[0].get('message', 'unknown')}")

        return data.get("data", {})

    # ── Inline fragments shared across queries ──────────────────
    _INLINE = """
                        ... on BasicObject {
                          id
                          entity_type
                        }
                        ... on AttackPattern { name }
                        ... on Campaign { name }
                        ... on Malware { name }
                        ... on Tool { name }
                        ... on Vulnerability { name }
                        ... on ThreatActorGroup { name }
                        ... on ThreatActorIndividual { name }
                        ... on IntrusionSet { name }
                        ... on Infrastructure { name }
                        ... on Indicator { name }
                        ... on Identity { name }
                        ... on CourseOfAction { name }
                        ... on Report { name }"""

    def list_entities_with_relations(
        self,
        entity_type: str,
        first: int = 50,
    ) -> List[Dict[str, Any]]:
        """List ALL entities of a type with their first-hop relationships.

        Uses cursor-based pagination to fetch everything.
        """
        query_name = self._get_query_name(entity_type)

        all_entities: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        page_size = min(first, 100)

        while True:
            after_clause = f', after: "{cursor}"' if cursor else ""
            gql = f"""
            {{
              {query_name}(first: {page_size}{after_clause}, orderBy: created_at, orderMode: desc) {{
                pageInfo {{
                  hasNextPage
                  endCursor
                }}
                edges {{
                  node {{
                    id
                    name
                    description
                    stixCoreRelationships(first: 30) {{
                      edges {{
                        node {{
                          id
                          relationship_type
                          confidence
                          from {{
{self._INLINE}
                          }}
                          to {{
{self._INLINE}
                          }}
                        }}
                      }}
                    }}
                  }}
                }}
              }}
            }}
            """
            try:
                result = self.query(gql)
            except Exception as exc:
                logger.warning("Failed to list %s (cursor=%s): %s", entity_type, cursor, exc)
                break

            data = result.get(query_name, {})
            edges = data.get("edges", [])
            page_info = data.get("pageInfo", {})

            for edge in edges:
                node = edge.get("node", {})
                relations = []
                for rel_edge in (
                    node.get("stixCoreRelationships", {}).get("edges", [])
                ):
                    rel = rel_edge.get("node", {})
                    from_obj = rel.get("from") or {}
                    to_obj = rel.get("to") or {}
                    relations.append({
                        "id": rel.get("id"),
                        "type": rel.get("relationship_type"),
                        "confidence": rel.get("confidence", 50),
                        "from_id": from_obj.get("id"),
                        "from_name": from_obj.get("name", "unknown"),
                        "from_type": from_obj.get("entity_type", "unknown"),
                        "to_id": to_obj.get("id"),
                        "to_name": to_obj.get("name", "unknown"),
                        "to_type": to_obj.get("entity_type", "unknown"),
                    })

                all_entities.append({
                    "id": node.get("id"),
                    "name": node.get("name", "unknown"),
                    "type": entity_type,
                    "description": (node.get("description") or ""),
                    "relations": relations,
                })

            logger.info(
                "Fetched page of %d %s (total %d so far)",
                len(edges), entity_type, len(all_entities),
            )

            if not page_info.get("hasNextPage") or not edges:
                break
            cursor = page_info.get("endCursor")

        return all_entities

    def _get_query_name(self, entity_type: str) -> str:
        """Map entity type to GraphQL query name."""
        mapping = {
            "Malware": "malwares",
            "Threat-Actor": "threatActorsGroup",
            "Attack-Pattern": "attackPatterns",
            "Tool": "tools",
            "Vulnerability": "vulnerabilities",
            "Campaign": "campaigns",
            "Intrusion-Set": "intrusionSets",
            "Indicator": "indicators",
            "Infrastructure": "infrastructures",
            "Report": "reports",
            "Course-Of-Action": "coursesOfAction",
        }
        return mapping.get(entity_type, entity_type.lower() + "s")

    def list_reports_with_objects(
        self,
        first: int = 50,
    ) -> List[Dict[str, Any]]:
        """List ALL reports with their contained STIX objects.

        Uses cursor-based pagination.
        """
        _INLINE_SIMPLE = """
                          ... on BasicObject { id entity_type }
                          ... on AttackPattern { name }
                          ... on Campaign { name }
                          ... on Malware { name }
                          ... on Tool { name }
                          ... on Vulnerability { name }
                          ... on ThreatActorGroup { name }
                          ... on ThreatActorIndividual { name }
                          ... on IntrusionSet { name }
                          ... on Infrastructure { name }
                          ... on Indicator { name }
                          ... on Identity { name }
                          ... on CourseOfAction { name }"""

        all_reports: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        page_size = min(first, 100)

        while True:
            after_clause = f', after: "{cursor}"' if cursor else ""
            gql = f"""
            {{
              reports(first: {page_size}{after_clause}, orderBy: created_at, orderMode: desc) {{
                pageInfo {{
                  hasNextPage
                  endCursor
                }}
                edges {{
                  node {{
                    id
                    name
                    description
                    published
                    content
                    objects(first: 200) {{
                      edges {{
                        node {{
                          ... on BasicObject {{
                            id
                            entity_type
                          }}
                          ... on AttackPattern {{ name }}
                          ... on Campaign {{ name }}
                          ... on Malware {{ name }}
                          ... on Tool {{ name }}
                          ... on Vulnerability {{ name }}
                          ... on ThreatActorGroup {{ name }}
                          ... on ThreatActorIndividual {{ name }}
                          ... on IntrusionSet {{ name }}
                          ... on Infrastructure {{ name }}
                          ... on Indicator {{ name }}
                          ... on Identity {{ name }}
                          ... on CourseOfAction {{ name }}
                          ... on StixCoreRelationship {{
                            id
                            relationship_type
                            confidence
                            from {{
{_INLINE_SIMPLE}
                            }}
                            to {{
{_INLINE_SIMPLE}
                            }}
                          }}
                        }}
                      }}
                    }}
                  }}
                }}
              }}
            }}
            """
            try:
                result = self.query(gql)
            except Exception as exc:
                logger.warning("Failed to list reports (cursor=%s): %s", cursor, exc)
                break

            data = result.get("reports", {})
            edges = data.get("edges", [])
            page_info = data.get("pageInfo", {})

            for edge in edges:
                node = edge.get("node", {})
                contained_objects = []
                contained_relations = []

                for obj_edge in node.get("objects", {}).get("edges", []):
                    obj = obj_edge.get("node", {})
                    etype = obj.get("entity_type", "")

                    if obj.get("relationship_type"):
                        from_obj = obj.get("from") or {}
                        to_obj = obj.get("to") or {}
                        contained_relations.append({
                            "id": obj.get("id"),
                            "type": obj.get("relationship_type"),
                            "confidence": obj.get("confidence", 50),
                            "from_id": from_obj.get("id"),
                            "from_name": from_obj.get("name", "unknown"),
                            "from_type": from_obj.get("entity_type", "unknown"),
                            "to_id": to_obj.get("id"),
                            "to_name": to_obj.get("name", "unknown"),
                            "to_type": to_obj.get("entity_type", "unknown"),
                        })
                    elif obj.get("name"):
                        contained_objects.append({
                            "id": obj.get("id"),
                            "name": obj["name"],
                            "type": etype,
                        })

                text_parts = []
                if node.get("description"):
                    text_parts.append(node["description"])
                if node.get("content"):
                    text_parts.append(node["content"])
                full_text = "\n\n".join(text_parts)

                all_reports.append({
                    "id": node.get("id"),
                    "name": node.get("name", "unknown"),
                    "description": (node.get("description") or "")[:1000],
                    "text": full_text,
                    "published": node.get("published"),
                    "objects": contained_objects,
                    "relations": contained_relations,
                })

            logger.info(
                "Fetched page of %d reports (total %d so far)",
                len(edges), len(all_reports),
            )

            if not page_info.get("hasNextPage") or not edges:
                break
            cursor = page_info.get("endCursor")

        return all_reports
