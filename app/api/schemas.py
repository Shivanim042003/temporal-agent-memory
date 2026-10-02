from datetime import datetime

from pydantic import BaseModel, Field


class MemoryQueryRequest(BaseModel):
    query: str = Field(
        min_length=1
    )

    source_id: str = Field(
        default="api-query",
        min_length=1,
    )


class MemoryQueryResponse(BaseModel):
    answer: str

    memories: list[dict]

    query_time: datetime | None