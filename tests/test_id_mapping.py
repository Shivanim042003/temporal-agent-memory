import pytest

from app.retrieval.id_mapping import MemoryIDMapping


def test_mapping_starts_empty():
    mapping = MemoryIDMapping()

    assert mapping.size == 0


def test_add_maps_vector_id_to_memory_id():
    mapping = MemoryIDMapping()

    mapping.add(
        vector_id=0,
        memory_id="mem_001",
    )

    assert mapping.size == 1
    assert mapping.get_memory_id(0) == "mem_001"


def test_add_maps_memory_id_to_vector_id():
    mapping = MemoryIDMapping()

    mapping.add(
        vector_id=7,
        memory_id="mem_007",
    )

    assert mapping.get_vector_id("mem_007") == 7


def test_missing_vector_id_returns_none():
    mapping = MemoryIDMapping()

    assert mapping.get_memory_id(999) is None


def test_missing_memory_id_returns_none():
    mapping = MemoryIDMapping()

    assert mapping.get_vector_id("does-not-exist") is None


def test_mapping_supports_multiple_memories():
    mapping = MemoryIDMapping()

    mapping.add(0, "mem_001")
    mapping.add(1, "mem_002")
    mapping.add(2, "mem_003")

    assert mapping.size == 3

    assert mapping.get_memory_id(0) == "mem_001"
    assert mapping.get_memory_id(1) == "mem_002"
    assert mapping.get_memory_id(2) == "mem_003"

    assert mapping.get_vector_id("mem_001") == 0
    assert mapping.get_vector_id("mem_002") == 1
    assert mapping.get_vector_id("mem_003") == 2


def test_duplicate_vector_id_is_rejected():
    mapping = MemoryIDMapping()

    mapping.add(0, "mem_001")

    with pytest.raises(
        ValueError,
        match="vector_id already exists",
    ):
        mapping.add(0, "mem_002")


def test_duplicate_memory_id_is_rejected():
    mapping = MemoryIDMapping()

    mapping.add(0, "mem_001")

    with pytest.raises(
        ValueError,
        match="memory_id already exists",
    ):
        mapping.add(1, "mem_001")


def test_negative_vector_id_is_rejected():
    mapping = MemoryIDMapping()

    with pytest.raises(
        ValueError,
        match="vector_id must be non-negative",
    ):
        mapping.add(-1, "mem_001")


def test_empty_memory_id_is_rejected():
    mapping = MemoryIDMapping()

    with pytest.raises(
        ValueError,
        match="memory_id must not be empty",
    ):
        mapping.add(0, "")


def test_remove_by_vector_id():
    mapping = MemoryIDMapping()

    mapping.add(5, "mem_005")

    removed = mapping.remove_by_vector_id(5)

    assert removed == "mem_005"
    assert mapping.size == 0
    assert mapping.get_memory_id(5) is None
    assert mapping.get_vector_id("mem_005") is None


def test_remove_by_memory_id():
    mapping = MemoryIDMapping()

    mapping.add(5, "mem_005")

    removed = mapping.remove_by_memory_id("mem_005")

    assert removed == 5
    assert mapping.size == 0
    assert mapping.get_memory_id(5) is None
    assert mapping.get_vector_id("mem_005") is None


def test_remove_missing_vector_id_returns_none():
    mapping = MemoryIDMapping()

    assert mapping.remove_by_vector_id(999) is None


def test_remove_missing_memory_id_returns_none():
    mapping = MemoryIDMapping()

    assert mapping.remove_by_memory_id("does-not-exist") is None

def test_save_creates_mapping_file(tmp_path):
    mapping = MemoryIDMapping()

    mapping.add(0, "mem_001")
    mapping.add(1, "mem_002")

    path = tmp_path / "memory_ids.json"

    mapping.save(path)

    assert path.exists()
    assert path.is_file()


def test_load_restores_mapping(tmp_path):
    mapping = MemoryIDMapping()

    mapping.add(0, "mem_001")
    mapping.add(1, "mem_002")

    path = tmp_path / "memory_ids.json"

    mapping.save(path)

    loaded = MemoryIDMapping.load(path)

    assert loaded.size == 2

    assert loaded.get_memory_id(0) == "mem_001"
    assert loaded.get_memory_id(1) == "mem_002"

    assert loaded.get_vector_id("mem_001") == 0
    assert loaded.get_vector_id("mem_002") == 1


def test_save_creates_parent_directories(tmp_path):
    mapping = MemoryIDMapping()

    mapping.add(0, "mem_001")

    path = (
        tmp_path
        / "indexes"
        / "nested"
        / "memory_ids.json"
    )

    mapping.save(path)

    assert path.exists()


def test_load_rejects_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.json"

    with pytest.raises(
        FileNotFoundError,
        match="memory ID mapping does not exist",
    ):
        MemoryIDMapping.load(path)    