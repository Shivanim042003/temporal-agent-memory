from datetime import datetime

from pydantic import BaseModel, Field

from app.memory.extraction import MemoryExtractionCandidate
from app.memory.models import Memory


class MemoryAgentState(BaseModel):
    """
    Shared state passed between LangGraph memory workflow nodes.
    """

    conversation: str = Field(min_length=1)

    source_id: str = Field(min_length=1)

    memory_keys: list[str] = Field(
        default_factory=list
    )

    extracted_memories: list[MemoryExtractionCandidate] = Field(
        default_factory=list
    )

    validated_memories: list[MemoryExtractionCandidate] = Field(
        default_factory=list
    )

    written_memories: list[Memory] = Field(
        default_factory=list
    )

    context_memories: list[Memory] = Field(
        default_factory=list
    )

    query_time: datetime | None = None

    error: str | None = None

    answer: str | None = None