"""
masked_scaler.py — StandardScaler that outputs 0 for chosen feature columns.

Used by the feature-masked detector sets (train_adversarial.py --mask-feature).
Every script scales through scaler.transform, so pickling this scaler into a
detector directory removes the column at training time and at every inference
call (ensemble.py, evaluate_detectors.py, the simulator), with no other code
changes. A detector cannot use a feature that is constant 0 in all its inputs.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class MaskedScaler(StandardScaler):
    masked_idx = ()

    @classmethod
    def from_fitted(cls, scaler, masked_names):
        """Copy a fitted StandardScaler and mask the named columns."""
        out = cls()
        out.__dict__.update(scaler.__dict__)
        out.masked_idx = tuple(list(scaler.feature_names_in_).index(n) for n in masked_names)
        out.masked_names = tuple(masked_names)
        return out

    def transform(self, X, copy=None):
        Z = np.array(super().transform(X, copy=copy), copy=True)
        if self.masked_idx:
            Z[:, list(self.masked_idx)] = 0.0
        return Z
