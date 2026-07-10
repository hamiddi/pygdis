import numpy as np


class LogisticMap:
    name = "Logistic"
    critical_value = 3.56995

    def __init__(self, parameter_values=None, iterations=5000, transient=2000, initial_value=0.1234567):
        self.parameter_values = np.linspace(2.5, 4.0, 101) if parameter_values is None else np.asarray(parameter_values, dtype=float)
        self.iterations = iterations
        self.transient = transient
        self.initial_value = initial_value

    def simulate(self, parameter):
        values = np.zeros(self.iterations, dtype=float)
        values[0] = self.initial_value
        for index in range(self.iterations - 1):
            values[index + 1] = parameter * values[index] * (1.0 - values[index])
        return values[self.transient:, None]

    def generate_sweep(self):
        return [self.simulate(float(p)) for p in self.parameter_values], self.parameter_values.copy()
