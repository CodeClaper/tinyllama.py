
from typing import List
from src.sequence import SequenceGroup


class Policy:

    def get_priority(
        self, 
        now: float, 
        seq_grop: SequenceGroup,
    ) -> float:
        raise NotImplementedError()


    def sort_by_priority(
        self,
        now: float,
        seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups, 
            key=lambda seq_grop: self.get_priority(now, seq_grop),
            reverse=True
        )


class FCFS(Policy):
    
    def get_priority(
        self, 
        now: float, 
        seq_grop: SequenceGroup
    ) -> float:
        return now - seq_grop.arrival_time



class PolicyFactory:

    _POLICY_REGISTRY = {
        "fcfs": FCFS
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

