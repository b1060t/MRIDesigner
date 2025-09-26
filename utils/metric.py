import numpy as np
import logging
logger = logging.getLogger(__name__)

class Metric:
    def __init__(self):
        self.metric = None

    def compute(self, data):
        self.metric = data

    def get(self, data):
        self.compute(data)
        return self.metric

class StrengthMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, field):
        super().compute(field)
        self.metric = np.nanmean(field)

class UniformityMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, field):
        super().compute(field)
        self.metric = (np.nanmax(field) - np.nanmin(field)) / np.nanmean(field) * 1e6

class SymmetricUniformityMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, field):
        super().compute(field)
        flipField = np.flip(field, axis=-1)
        flipField = flipField - field
        self.metric = (np.nanmax(flipField) - np.nanmin(flipField)) / np.nanmean(field) * 1e6

class LengthMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, design):
        super().compute(design)
        zpos = design.getPos()[:, 2]
        min_z = np.nanmin(zpos)
        max_z = np.nanmax(zpos)
        self.metric = max_z - min_z + design.meta['size']

class MagNumMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, design):
        super().compute(design)
        self.metric = design.getPos().shape[0]

class WeightMetric(Metric):
    def __init__(self):
        super().__init__()

    def compute(self, weight):
        super().compute(weight)
        self.metric = weight

Metric_LUT = {
    'strength': StrengthMetric,
    'uniformity': UniformityMetric,
    'symuniformity': SymmetricUniformityMetric,
    'length': LengthMetric,
    'magNum': MagNumMetric,
    'weight': WeightMetric,
    'maxB': Metric,
}

class MetricBuilder:
    def __init__(self, metric_config):
        self.equation = metric_config['equation']
        self.equation = ''.join(self.equation)
        self.vars = metric_config['vars']
        self.allowed = {
            'np': np,
        }
    
    def get(self, data):
        vars = {}
        for var in self.vars:
            var_type = var['type']
            var_data = var['data']
            if var_type in Metric_LUT:
                metric = Metric_LUT[var_type]()
                var_data = data[var_data]
                metric = metric.get(var_data)
                logger.debug(f"Metric {var_type} - Value: {metric}")
                if metric is None:
                    raise ValueError(f"Metric {var_type} computation failed.")
                vars[var_type] = metric
            else:
                raise ValueError(f"Unknown metric type: {var_type}")
        allowed = self.allowed.copy()
        allowed.update(vars)
        logger.debug(f"Evaluating metric with allowed variables: {allowed}")
        try:
            return eval(self.equation, {"__builtins__": None}, allowed)
        except Exception as e:
            logger.error(f"Error evaluating metric equation: {e}")
            return None