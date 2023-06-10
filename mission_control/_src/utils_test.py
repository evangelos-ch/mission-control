from .utils import unique_id


def test_unique_id():
    assert len(unique_id()) == 36
    assert unique_id() != unique_id()
