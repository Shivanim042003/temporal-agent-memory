from datetime import datetime, timezone

import pytest

from app.agent.reasoner import StaticAgentReasoner
from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
)


def make_memory(
    *,
    value: str = "Python",
) -> Memory:
    return Memory(
        memory_id="memory-1",
        memory_key="user:language",
        subject="user",
        attribute="language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        source_type=SourceType.CONVERSATION,
        source_id="conversation-1",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=0.95,
    )


def test_reasoner_returns_message_when_no_memories_exist():
    reasoner = StaticAgentReasoner()

    answer = reasoner.answer(
        "What language do I use?",
        memories=[],
    )

    assert answer == "No relevant memories found."


def test_reasoner_includes_memory_context():
    reasoner = StaticAgentReasoner()

    answer = reasoner.answer(
        "What language do I use?",
        memories=[
            make_memory(),
        ],
    )

    assert "What language do I use?" in answer
    assert "user language: Python" in answer


def test_reasoner_includes_multiple_memories():
    reasoner = StaticAgentReasoner()

    answer = reasoner.answer(
        "What technologies do I use?",
        memories=[
            make_memory(value="Python"),
            Memory(
                memory_id="memory-2",
                memory_key="user:language",
                subject="user",
                attribute="language",
                value="Java",
                memory_type=MemoryType.SKILL,
                valid_from=datetime(
                    2026,
                    1,
                    1,
                    tzinfo=timezone.utc,
                ),
                source_type=SourceType.CONVERSATION,
                source_id="conversation-2",
                evidence_type=EvidenceType.EXPLICIT,
                confidence=0.9,
            ),
        ],
    )

    assert "Python" in answer
    assert "Java" in answer


def test_reasoner_rejects_empty_query():
    reasoner = StaticAgentReasoner()

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        reasoner.answer(
            "",
            memories=[],
        )


def test_reasoner_rejects_whitespace_query():
    reasoner = StaticAgentReasoner()

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        reasoner.answer(
            "   ",
            memories=[],
        )