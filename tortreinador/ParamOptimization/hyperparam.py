from dataclasses import dataclass
import random
from typing import List, Any
import math

@dataclass
class HyperParam:
    def sample(self):
        raise NotImplementedError

@dataclass
class IntParam(HyperParam):
    low: int
    high: int

    def sample(self):
        return random.randint(self.low, self.high)

@dataclass
class FloatParam(HyperParam):
    low: float
    high: float
    def sample(self):
        return random.uniform(self.low, self.high)

@dataclass
class LogFloatParam(HyperParam):
    low: float
    high: float

    def sample(self):
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(random.uniform(log_low, log_high))

@dataclass
class ChoiceParam(HyperParam):
    choices: List[Any]

    def sample(self):
        return random.choice(self.choices)