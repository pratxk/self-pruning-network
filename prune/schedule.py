from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def target(self, step: int) -> float:
        """Sparsity the model must have at this step."""

    @abstractmethod
    def is_prune_step(self, step: int) -> bool: ...


class CubicSchedule(Schedule):
    """Zhu & Gupta 2017: s_t = s_f (1 - (1 - p)^3), p = progress in [0, 1].
    Prunes fast while redundancy is high, slowly near the target."""

    def __init__(self, s_final: float, start_step: int, end_step: int, every: int = 100):
        if end_step <= start_step:
            raise ValueError("end_step must be after start_step")
        self.s_final, self.start, self.end, self.every = s_final, start_step, end_step, every

    def target(self, step):
        p = min(max((step - self.start) / (self.end - self.start), 0.0), 1.0)
        return self.s_final * (1 - (1 - p) ** 3)

    def is_prune_step(self, step):
        in_ramp = self.start <= step <= self.end and (step - self.start) % self.every == 0
        return in_ramp or step == self.end  # always land exactly on s_final


class OneShotSchedule(Schedule):
    def __init__(self, s_final: float, at_step: int):
        self.s_final, self.at = s_final, at_step

    def target(self, step):
        return self.s_final if step >= self.at else 0.0

    def is_prune_step(self, step):
        return step == self.at


class ConstantSchedule(Schedule):
    """Dense runs."""

    def __init__(self, sparsity: float = 0.0):
        self.sparsity = sparsity

    def target(self, step):
        return self.sparsity

    def is_prune_step(self, step):
        return False
