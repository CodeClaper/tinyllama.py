import enum
import uuid

class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()

class PreemptionMode(enum.Enum):
    """
    Preemption mode.
    
    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap the back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequence and recompute
    them when the sequences are resumed, treating the sequences as new prompts.
    """
    SWAP      = enum.auto()
    RECOMPUTE = enum.auto()

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
