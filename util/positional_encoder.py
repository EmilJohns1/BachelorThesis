import numpy as np

class PositionalEncoder:
    def __init__(self, d_model=4):
        self.d_model = d_model

    def encode_scalar(self, x):
        pe = np.zeros(self.d_model)
        for i in range(0, self.d_model, 2):
            div_term = np.exp(-np.log(10000.0) * i / self.d_model)
            pe[i] = np.sin(x * div_term)
            if i + 1 < self.d_model:
                pe[i + 1] = np.cos(x * div_term)
        return pe

    def encode(self, state):
        encoded = [self.encode_scalar(x) for x in state]
        return np.concatenate(encoded)