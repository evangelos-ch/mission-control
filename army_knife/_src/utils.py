import uuid


def unique_id() -> str:
    return str(uuid.uuid4())
