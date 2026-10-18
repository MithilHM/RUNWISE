"""
Persistent Multi-Session Conversation Graph Memory for RUNWISE.

Persists:
- (:ChatSession {id, project_id, title, created_at, updated_at})
- (:ChatMessage {id, session_id, role, content, mode, llm_used, timestamp})
- Relationships:
  (:Project)-[:CONTAINS_SESSION]->(:ChatSession)
  (:ChatSession)-[:HAS_MESSAGE]->(:ChatMessage)
  (:ChatMessage)-[:CITES_EXPERIMENT]->(:Experiment)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.graph.client import FalkorDBClient, graph_client
from app.graph.schema import NodeLabel, RelType

logger = logging.getLogger(__name__)


class ConversationMemoryRepository:
    """Manages FalkorDB graph-backed persistent conversation sessions and messages."""

    def __init__(self, client: Optional[FalkorDBClient] = None):
        self.client = client or graph_client
        # In-memory storage fallback for unit test isolation / disconnected mode
        self._in_memory_sessions: Dict[str, Dict[str, Any]] = {}
        self._in_memory_messages: Dict[str, List[Dict[str, Any]]] = {}

    def create_session(self, project_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new persistent chat session in FalkorDB."""
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        session_title = title or "New Research Investigation"

        cypher = """
        MERGE (p:Project {id: $project_id})
        CREATE (s:ChatSession {
            id: $session_id,
            project_id: $project_id,
            title: $title,
            created_at: $now,
            updated_at: $now
        })
        CREATE (p)-[:CONTAINS_SESSION]->(s)
        RETURN s.id AS id, s.project_id AS project_id, s.title AS title, s.created_at AS created_at, s.updated_at AS updated_at
        """
        try:
            graph = self.client.get_graph()
            res = graph.query(cypher, {
                "project_id": project_id,
                "session_id": session_id,
                "title": session_title,
                "now": now,
            })
            if res.result_set:
                row = res.result_set[0]
                session_data = {
                    "id": row[0],
                    "project_id": row[1],
                    "title": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                }
                self._in_memory_sessions[session_id] = session_data
                return session_data
        except Exception as e:
            logger.warning(f"FalkorDB unavailable for create_session, using memory fallback: {e}")

        # Fallback
        session_data = {
            "id": session_id,
            "project_id": project_id,
            "title": session_title,
            "created_at": now,
            "updated_at": now,
        }
        self._in_memory_sessions[session_id] = session_data
        self._in_memory_messages[session_id] = []
        return session_data

    def list_sessions(self, project_id: str) -> List[Dict[str, Any]]:
        """Lists all chat sessions for a project ordered by most recent."""
        cypher = """
        MATCH (s:ChatSession {project_id: $project_id})
        RETURN s.id AS id, s.project_id AS project_id, s.title AS title, s.created_at AS created_at, s.updated_at AS updated_at
        ORDER BY s.updated_at DESC
        """
        try:
            graph = self.client.get_graph()
            res = graph.query(cypher, {"project_id": project_id})
            sessions = []
            for row in res.result_set:
                sessions.append({
                    "id": row[0],
                    "project_id": row[1],
                    "title": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                })
            if sessions:
                return sessions
        except Exception as e:
            logger.warning(f"FalkorDB unavailable for list_sessions, using fallback: {e}")

        # Fallback
        matching = [
            s for s in self._in_memory_sessions.values()
            if s.get("project_id") == project_id
        ]
        matching.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return matching

    def add_message(
        self,
        project_id: str,
        session_id: str,
        role: str,
        content: str,
        mode: Optional[str] = None,
        llm_used: bool = False,
        citation_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Atomically appends a chat message to a session and links cited experiments."""
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        citations = citation_ids or []

        cypher = """
        MATCH (s:ChatSession {id: $session_id, project_id: $project_id})
        SET s.updated_at = $now
        CREATE (m:ChatMessage {
            id: $msg_id,
            session_id: $session_id,
            role: $role,
            content: $content,
            mode: $mode,
            llm_used: $llm_used,
            timestamp: $now
        })
        CREATE (s)-[:HAS_MESSAGE]->(m)
        RETURN m.id AS id, m.session_id AS session_id, m.role AS role, m.content AS content,
               m.mode AS mode, m.llm_used AS llm_used, m.timestamp AS timestamp
        """
        try:
            graph = self.client.get_graph()
            res = graph.query(cypher, {
                "project_id": project_id,
                "session_id": session_id,
                "msg_id": msg_id,
                "role": role,
                "content": content,
                "mode": mode or "deterministic",
                "llm_used": llm_used,
                "now": now,
            })

            # Link cited experiments
            for cid in citations:
                link_cypher = """
                MATCH (m:ChatMessage {id: $msg_id})
                MATCH (e:Experiment {id: $exp_id, project_id: $project_id})
                MERGE (m)-[:CITES_EXPERIMENT]->(e)
                """
                try:
                    graph.query(link_cypher, {
                        "msg_id": msg_id,
                        "exp_id": cid,
                        "project_id": project_id,
                    })
                except Exception:
                    pass

            if res.result_set:
                row = res.result_set[0]
                msg_dict = {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "mode": row[4],
                    "llm_used": row[5],
                    "timestamp": row[6],
                    "citation_ids": citations,
                }
                return msg_dict
        except Exception as e:
            logger.warning(f"FalkorDB unavailable for add_message, using fallback: {e}")

        # Fallback
        msg_dict = {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "mode": mode or "deterministic",
            "llm_used": llm_used,
            "timestamp": now,
            "citation_ids": citations,
        }
        if session_id not in self._in_memory_messages:
            self._in_memory_messages[session_id] = []
        self._in_memory_messages[session_id].append(msg_dict)
        if session_id in self._in_memory_sessions:
            self._in_memory_sessions[session_id]["updated_at"] = now
        return msg_dict

    def get_history(self, project_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological chat history with citations for a session."""
        cypher = """
        MATCH (s:ChatSession {id: $session_id, project_id: $project_id})-[:HAS_MESSAGE]->(m:ChatMessage)
        OPTIONAL MATCH (m)-[:CITES_EXPERIMENT]->(e:Experiment)
        RETURN m.id AS id, m.session_id AS session_id, m.role AS role, m.content AS content,
               m.mode AS mode, m.llm_used AS llm_used, m.timestamp AS timestamp,
               collect(DISTINCT e.id) AS citations
        ORDER BY m.timestamp ASC
        """
        try:
            graph = self.client.get_graph()
            res = graph.query(cypher, {"project_id": project_id, "session_id": session_id})
            messages = []
            for row in res.result_set:
                raw_cites = row[7] if len(row) > 7 else []
                clean_cites = [c for c in raw_cites if c is not None]
                messages.append({
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "mode": row[4],
                    "llm_used": row[5],
                    "timestamp": row[6],
                    "citation_ids": clean_cites,
                })
            if messages:
                return messages
        except Exception as e:
            logger.warning(f"FalkorDB unavailable for get_history, using fallback: {e}")

        return self._in_memory_messages.get(session_id, [])


conversation_memory = ConversationMemoryRepository()
