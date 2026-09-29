from datetime import datetime, timezone
from enum import StrEnum

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


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

    def to_embedding_text(self) -> str:
        return (
            f"subject: {self.subject}\n"
            f"attribute: {self.attribute}\n"
            f"value: {self.value}\n"
            f"type: {self.memory_type.value}"
        )


def make_memory(**overrides):
    values = {
        "memory_id": "mem_1",
        "memory_key": "user.programming_language",
        "subject": "user",
        "attribute": "programming_language",
        "value": "Python",
        "memory_type": MemoryType.SKILL,
        "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_1",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.95,
    }

    values.update(overrides)

    return Memory(**values)


def test_memory_accepts_valid_data():
    memory = make_memory()

    assert memory.memory_id == "mem_1"
    assert memory.memory_key == "user.programming_language"
    assert memory.subject == "user"
    assert memory.attribute == "programming_language"
    assert memory.value == "Python"
    assert memory.memory_type == MemoryType.SKILL
    assert memory.status == MemoryStatus.ACTIVE


def test_memory_defaults_recorded_at():
    before = datetime.now(timezone.utc)

    memory = make_memory()

    after = datetime.now(timezone.utc)

    assert before <= memory.recorded_at <= after


def test_memory_normalizes_datetime_to_utc():
    memory = make_memory(
        valid_from=datetime(
            2026,
            1,
            1,
            12,
            tzinfo=timezone.utc,
        )
    )

    assert memory.valid_from.tzinfo == timezone.utc


def test_memory_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        make_memory(
            valid_from=datetime(2026, 1, 1)
        )


def test_memory_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        make_memory(confidence=1.1)

    with pytest.raises(ValidationError):
        make_memory(confidence=-0.1)


def test_memory_rejects_empty_memory_id():
    with pytest.raises(ValidationError):
        make_memory(memory_id="")


def test_memory_rejects_empty_memory_key():
    with pytest.raises(ValidationError):
        make_memory(memory_key="")


def test_memory_rejects_empty_subject():
    with pytest.raises(ValidationError):
        make_memory(subject="")


def test_memory_rejects_empty_attribute():
    with pytest.raises(ValidationError):
        make_memory(attribute="")


def test_memory_rejects_empty_value():
    with pytest.raises(ValidationError):
        make_memory(value="")


def test_memory_rejects_invalid_temporal_interval():
    with pytest.raises(ValidationError):
        make_memory(
            valid_to=datetime(
                2025,
                12,
                31,
                tzinfo=timezone.utc,
            )
        )


def test_historical_memory_requires_valid_to():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.HISTORICAL,
            valid_to=None,
        )


def test_superseded_memory_requires_valid_to():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.SUPERSEDED,
            valid_to=None,
        )


def test_memory_cannot_supersede_itself():
    with pytest.raises(ValidationError):
        make_memory(
            supersedes_id="mem_1",
        )


def test_memory_cannot_be_consolidated_into_itself():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.CONSOLIDATED,
            canonical_memory_id="mem_1",
        )


def test_consolidated_memory_requires_canonical_memory_id():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.CONSOLIDATED,
            canonical_memory_id=None,
        )


def test_non_consolidated_memory_cannot_have_canonical_memory_id():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.ACTIVE,
            canonical_memory_id="canonical_1",
        )


def test_consolidated_memory_can_have_open_ended_interval():
    memory = make_memory(
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id="mem_canonical",
        valid_to=None,
    )

    assert memory.status == MemoryStatus.CONSOLIDATED
    assert memory.valid_to is None


def test_discarded_memory_preserves_original_interval():
    memory = make_memory(
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        status=MemoryStatus.DISCARDED,
    )

    assert memory.status == MemoryStatus.DISCARDED
    assert memory.valid_from == datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )
    assert memory.valid_to == datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )


def test_discarded_memory_can_have_open_ended_interval():
    memory = make_memory(
        valid_to=None,
        status=MemoryStatus.DISCARDED,
    )

    assert memory.status == MemoryStatus.DISCARDED
    assert memory.valid_to is None


def test_to_embedding_text():
    memory = make_memory(
        subject="user",
        attribute="programming_language",
        value="Python",
    )

    assert (
        memory.to_embedding_text()
        == "subject: user\n"
        "attribute: programming_language\n"
        "value: Python\n"
        "type: skill"
    )


def test_to_embedding_text_is_deterministic():
    memory = make_memory()

    first = memory.to_embedding_text()
    second = memory.to_embedding_text()

    assert first == second


def test_to_embedding_text_uses_memory_fields():
    memory = make_memory(
        subject="user",
        attribute="backend_framework",
        value="FastAPI",
    )

    assert (
        memory.to_embedding_text()
        == "subject: user\n"
        "attribute: backend_framework\n"
        "value: FastAPI\n"
        "type: skill"
    )