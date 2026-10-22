import uuid

def random_uuid() -> str:
    """Returns random uuid str. """
    return str(uuid.uuid4().hex)
