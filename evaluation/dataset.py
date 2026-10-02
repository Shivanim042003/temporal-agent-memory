from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    memory_key: str
    query_time: datetime
    expected_value: str


DATASET = [
    EvaluationCase(
        query="What language was I using?",
        memory_key="user:language",
        query_time=datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        expected_value="Node.js",
    ),
    EvaluationCase(
        query="What language was I using?",
        memory_key="user:language",
        query_time=datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
        expected_value="Go",
    ),
    EvaluationCase(
        query="What language am I using now?",
        memory_key="user:language",
        query_time=datetime(
            2026,
            10,
            1,
            tzinfo=timezone.utc,
        ),
        expected_value="Python",
    ),
]