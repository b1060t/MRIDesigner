import numpy as np

class Metric:
    def __init__(self, lo=None, hi=None, weight=1.0):
        self.metric = None

    def compute(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

class StrengthMetric(Metric):
    def __init__(self):
        super().__init__()