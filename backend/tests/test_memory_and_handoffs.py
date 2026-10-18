"""
Tests for Persistent Conversation Memory, 3-Tier Memory Taxonomy, and Agent Handoffs.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.graph.conversation_memory import conversation_memory
from app.services.memory_service import memory_service
from app.graph.runtime_queries import runtime_graph_service


@pytest.fixture
def client():
    return TestClient(app)


def test_conversation_memory_crud():
    """Test creating sessions, adding messages, and retrieving history."""
    proj = "test_memory_proj"
    sess = conversation_memory.create_session(proj, title="Fraud Invalidation Investigation")
    assert sess["id"].startswith("sess_")
    assert sess["title"] == "Fraud Invalidation Investigation"

    # Add user message
    msg1 = conversation_memory.add_message(
        project_id=proj,
        session_id=sess["id"],
        role="user",
        content="Which runs are invalid?",
    )
    assert msg1["role"] == "user"

    # Add assistant response with citations
    msg2 = conversation_memory.add_message(
        project_id=proj,
        session_id=sess["id"],
        role="assistant",
        content="Runs exp_001 and exp_002 are invalid due to data leakage.",
        citation_ids=["exp_001", "exp_002"],
    )
    assert msg2["role"] == "assistant"
    assert "exp_001" in msg2["citation_ids"]

    # Retrieve history
    history = conversation_memory.get_history(proj, sess["id"])
    assert len(history) >= 2
    assert history[0]["content"] == "Which runs are invalid?"
    assert history[1]["content"] == "Runs exp_001 and exp_002 are invalid due to data leakage."


def test_memory_service_breakdown():
    """Test 3-tier memory service (Episodic, Semantic, Procedural)."""
    mem = memory_service.get_memory_breakdown("project_demo")
    assert "episodic" in mem
    assert "semantic" in mem
    assert "procedural" in mem

    # Episodic tier
    assert mem["episodic"]["tier"] == "episodic"
    assert "total_runs" in mem["episodic"]
    assert "failure_episodes" in mem["episodic"]

    # Semantic tier
    assert mem["semantic"]["tier"] == "semantic"
    assert "regimes" in mem["semantic"]
    assert "regime_similarities" in mem["semantic"]

    # Procedural tier
    assert mem["procedural"]["tier"] == "procedural"
    assert "rules" in mem["procedural"]
    assert "ranking_weights" in mem["procedural"]


def test_agent_handoff_recording():
    """Test recording and retrieving agent handoffs."""
    run_id = "test_run_handoff_01"
    h1 = runtime_graph_service.record_agent_handoff(
        from_agent="CandidatePlanner",
        to_agent="LayaEvaluator",
        run_id=run_id,
        step=1,
        state="CANDIDATE_EVALUATION",
        metadata={"candidate_count": 3},
    )
    assert h1["from_agent"] == "CandidatePlanner"
    assert h1["to_agent"] == "LayaEvaluator"

    h2 = runtime_graph_service.record_agent_handoff(
        from_agent="LayaEvaluator",
        to_agent="SandboxedWorker",
        run_id=run_id,
        step=1,
        state="EXECUTING_TRIAL",
        metadata={"candidate_id": "cand_exploit"},
    )
    assert h2["from_agent"] == "LayaEvaluator"
    assert h2["to_agent"] == "SandboxedWorker"

    handoffs = runtime_graph_service.get_agent_handoffs(run_id)
    assert len(handoffs) >= 2
    assert handoffs[0]["from_agent"] == "CandidatePlanner"
    assert handoffs[1]["from_agent"] == "LayaEvaluator"


def test_api_session_and_memory_endpoints(client):
    """Test REST endpoints for chat sessions and 3-tier memory."""
    # List sessions
    res_list = client.get("/api/v1/projects/project_demo/sessions")
    assert res_list.status_code == 200

    # Create session
    res_create = client.post(
        "/api/v1/projects/project_demo/sessions",
        json={"title": "Interactive Research Session"},
    )
    assert res_create.status_code == 201
    sess_id = res_create.json()["id"]

    # Get messages
    res_msgs = client.get(f"/api/v1/projects/project_demo/sessions/{sess_id}/messages")
    assert res_msgs.status_code == 200

    # Query within session
    res_query = client.post(
        f"/api/v1/projects/project_demo/sessions/{sess_id}/query",
        json={"question": "Which runs performed best?"},
    )
    assert res_query.status_code == 200
    assert "answer" in res_query.json()

    # Verify message was persisted
    res_msgs_after = client.get(f"/api/v1/projects/project_demo/sessions/{sess_id}/messages")
    assert res_msgs_after.status_code == 200
    assert len(res_msgs_after.json()) >= 2

    # Query 3-tier memory
    res_mem = client.get("/api/v1/projects/project_demo/memory")
    assert res_mem.status_code == 200
    assert "episodic" in res_mem.json()
    assert "semantic" in res_mem.json()
    assert "procedural" in res_mem.json()
