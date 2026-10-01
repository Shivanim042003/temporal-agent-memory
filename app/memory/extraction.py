from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
    TimePrecision,
)


class ExtractionEvidence(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class MemoryExtractionCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)

    memory_type: MemoryType

    valid_from: datetime
    valid_to: datetime | None = None

    confidence: float = Field(ge=0.0, le=1.0)

    evidence: ExtractionEvidence

    source_id: str = Field(min_length=1)

    @field_validator("valid_from", "valid_to")
    @classmethod
    def normalize_to_utc(cls, value: datetime | None):
        if value is None:
            return value

        if value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")

        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_temporal_interval(self):
        if (
            self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValueError(
                "valid_to must be later than valid_from"
            )

        return self

    def to_memory(
        self,
        *,
        memory_id: str,
        source_type: SourceType = SourceType.CONVERSATION,
        precision: TimePrecision = TimePrecision.EXACT,
    ) -> Memory:
        evidence_type = (
            EvidenceType.EXPLICIT
            if self.evidence == ExtractionEvidence.EXPLICIT
            else EvidenceType.INFERRED
        )

        return Memory(
            memory_id=memory_id,
            memory_key=f"{self.subject}:{self.attribute}",
            subject=self.subject,
            attribute=self.attribute,
            value=self.value,
            memory_type=self.memory_type,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            precision=precision,
            source_type=source_type,
            source_id=self.source_id,
            evidence_type=evidence_type,
            confidence=self.confidence,
        )


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    memories: list[MemoryExtractionCandidate] = Field(
        default_factory=list
    )