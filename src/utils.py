import uuid

class Counter:
    def __init__(self, start: int = 0) -> None:
        self._counter = start;

    def __next__(self) -> int:
        id = self._counter
        self._counter += 1
        return id

    def reset(self) -> None:
        self._counter = 0

def random_uuid() -> str:
    """Returns random uuid str. """
    return str(uuid.uuid4().hex)
