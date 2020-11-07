from typing import NamedTuple, Sequence

import numpy as np
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.trajectory import Trajectory


class Transition(NamedTuple):
    s: np.ndarray
    a: np.ndarray
    s_next: np.ndarray


class SimpleBuffer:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self.clear()

    def add(self, transition: Transition):
        self._store.append(obj=transition)

    def sample(self, n: int):
        return self._store.sample(batch_size=n)

    def clear(self):
        self._store = CircularList(max_size=self._capacity)


class CircularList:
    def __init__(self, max_size):
        self._buffer = [None] * max_size
        self._cur_head = 0
        self._cur_size = 0

    def append(self, obj):
        self._buffer[self._cur_head] = obj
        self._cur_size = min(self._cur_size + 1, len(self._buffer))
        self._cur_head = (self._cur_head + 1) % len(self._buffer)

    def sample(self, batch_size):
        assert self._cur_size > 0, "Is empty"
        indices = np.random.choice(self._cur_size, size=batch_size)
        samples = [self._buffer[i] for i in indices]
        return to_formatted_trajectory(samples)


def to_formatted_trajectory(samples: Sequence[Transition]) -> Trajectory:
    s, a, s_next = zip(*samples)
    two_cols = np.ones((len(s), 2))
    return Trajectory(
        step_type=StepType.MID * two_cols,
        observation=np.stack((s, s_next), axis=1),
        action=np.stack((a, np.NaN * np.ones_like(a)), axis=1),
        policy_info=(),
        next_step_type=StepType.MID * two_cols,
        reward=np.NaN * two_cols,
        discount=0.99 * two_cols
    )
