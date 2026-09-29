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
    CONSOLIDATED = "consolidated"
    DISCARDED = "discarded"


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
    canonical_memory_id: str | None = None

    @field_validator("valid_from", "valid_to", "recorded_at")
    @classmethod
    def normalize_to_utc(cls, value: datetime | None):
        if value is None:
            return value

        if value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")

        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_temporal_interval(self):
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be later than valid_from")

        return self

    @model_validator(mode="after")
    def validate_status_consistency(self):
        if (
            self.status in {
                MemoryStatus.HISTORICAL,
                MemoryStatus.SUPERSEDED,
            }
            and self.valid_to is None
        ):
            raise ValueError(
                "HISTORICAL/SUPERSEDED memory must have valid_to"
            )

        return self

    @model_validator(mode="after")
    def validate_supersedes_id(self):
        if self.supersedes_id == self.memory_id:
            raise ValueError("memory cannot supersede itself")

        return self

    @model_validator(mode="after")
    def validate_canonical_memory_id(self):
        if self.canonical_memory_id == self.memory_id:
            raise ValueError("memory cannot be canonical for itself")

        has_canonical = self.canonical_memory_id is not None
        is_consolidated = self.status == MemoryStatus.CONSOLIDATED

        if has_canonical and not is_consolidated:
            raise ValueError(
                "a memory with canonical_memory_id set must have "
                "status CONSOLIDATED"
            )

        if is_consolidated and not has_canonical:
            raise ValueError(
                "a CONSOLIDATED memory must have canonical_memory_id set"
            )

        return self