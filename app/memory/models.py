from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    EVENT = "event"
    SKILL = "skill"
    RELATIONSHIP = "relationship"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"
    UNCERTAIN = "uncertain"


class EvidenceType(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    SYSTEM_GENERATED = "system_generated"


class SourceType(StrEnum):
    CONVERSATION = "conversation"
    SYSTEM = "system"
    IMPORT = "import"


class TimePrecision(StrEnum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    EXACT = "exact"


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str = Field(min_length=1)
    memory_key: str = Field(min_length=1)

    subject: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)

    memory_type: MemoryType

    valid_from: datetime
    valid_to: datetime | None = None
    precision: TimePrecision = TimePrecision.EXACT

    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    source_type: SourceType
    source_id: str = Field(min_length=1)
    evidence_type: EvidenceType

    confidence: float = Field(ge=0.0, le=1.0)

    status: MemoryStatus = MemoryStatus.ACTIVE

    supersedes_id: str | None = None

    @field_validator("valid_from", "valid_to", "recorded_at")
    @classmethod
    def require_timezone_aware_datetime(cls, value: datetime | None):
        if value is not None and value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_temporal_interval(self):
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be later than valid_from")
        return self

    @model_validator(mode="after")
    def validate_active_memory(self):
        if (
            self.status == MemoryStatus.ACTIVE
            and self.valid_to is not None
            and self.valid_to <= datetime.now(timezone.utc)
        ):
            raise ValueError(
                "ACTIVE memory cannot have a valid_to in the past"
            )
        return self
