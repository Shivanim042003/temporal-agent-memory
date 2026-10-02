from datetime import datetime, timezone
from pathlib import Path

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
)
from app.storage.database import (
    initialize_database,
)
from app.storage.memory_repository import (
    insert,
)


def create_evaluation_database(
    db_path: Path,
) -> None:
    """
    Create a deterministic database containing the
    temporal memories used by the evaluation benchmark.

    The timeline is:

        Node.js  [2026-01-01, 2026-06-01)
        Go       [2026-06-01, 2026-09-01)
        Python   [2026-09-01, infinity)
    """

    initialize_database(
        db_path
    )

    memories = [
        Memory(
            memory_id="evaluation-language-node",
            memory_key="user:language",
            subject="user",
            attribute="language",
            value="Node.js",
            memory_type=MemoryType.SKILL,
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            source_type=SourceType.IMPORT,
            source_id="evaluation-fixture",
            evidence_type=EvidenceType.EXPLICIT,
            confidence=1.0,
        ),
        Memory(
            memory_id="evaluation-language-go",
            memory_key="user:language",
            subject="user",
            attribute="language",
            value="Go",
            memory_type=MemoryType.SKILL,
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
            source_type=SourceType.IMPORT,
            source_id="evaluation-fixture",
            evidence_type=EvidenceType.EXPLICIT,
            confidence=1.0,
        ),
        Memory(
            memory_id="evaluation-language-python",
            memory_key="user:language",
            subject="user",
            attribute="language",
            value="Python",
            memory_type=MemoryType.SKILL,
            valid_from=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
            source_type=SourceType.IMPORT,
            source_id="evaluation-fixture",
            evidence_type=EvidenceType.EXPLICIT,
            confidence=1.0,
        ),
    ]

    for memory in memories:
        insert(
            memory,
            db_path,
        )