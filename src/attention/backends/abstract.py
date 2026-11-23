
from dataclasses import dataclass, fields
from typing import Dict, Any

@dataclass
class AttentionMetadata:

    def asdict_zero_copy(self) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deep copy."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
