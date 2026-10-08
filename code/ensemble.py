"""
ensemble.py — Multi-model ensemble IDS for adversarial RL thesis.

Implements the ensemble defense layer from the thesis proposal:
three structurally diverse detectors (XGBoost, 1D-CNN, Random Forest)
combined under three aggregation rules:

  majority  : alert if >= 2 of 3 detectors flag the flow as intrusion
  unanimous : alert only if all 3 detectors flag the flow
  weighted  : alert if sum(w_i * confidence_i) >= threshold

Detector interface
------------------
`BaseDetector.predict(features) -> (label, confidence)` where
label is "scanning" or "normal" and confidence is in [0, 1].

The mock detectors below simulate architecturally diverse models with
different sensitivities so voting logic and DDQN training can be
exercised without the Pi or trained weights. Swap them for real models
by subclassing BaseDetector and loading ONNX/TFLite/sklearn artifacts
in `load()`.

Usage:
    from ensemble import EnsembleIDS, MockXGBoostDetector, MockCNNDetector, MockRFDetector

    ens = EnsembleIDS(voting="majority")
    result = ens.classify(features)   # features: (154,) array-like
    print(result["label"], result["votes"])
"""

import joblib
import numpy as np
import os
from abc import ABC, abstractmethod

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_DIR = "models"
VOTING_RULES = ("majority", "unanimous", "weighted")

# ── Detector Interface ─────────────────────────────────────────────────────
class BaseDetector(ABC):
    def __init__(self, model_name):
        self.name = model_name
        self.model = None
        self.scaler = None
        self.load()

    def load(self):
        model_path = os.path.join(MODEL_DIR, f"{self.name}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"[Ensemble] Loaded real {self.name} model.")
        else:
            print(f"[Ensemble] WARNING: {self.name} files not found. Using Mock mode.")

    @abstractmethod
    def predict(self, features):
        pass

# ── Real XGBoost Implementation ────────────────────────────────────────────
class XGBoostDetector(BaseDetector):
    def __init__(self):
        super().__init__("xgboost")

    def predict(self, features):
        if self.model is None: # Fallback to mock logic if file missing
            return ("normal", 0.5)
            
        # 1. Reshape and Scale
        feats_reshaped = features.reshape(1, -1)
        feats_scaled = self.scaler.transform(feats_reshaped)
        
        # 2. Predict Probability
        # Real models return [P(Benign), P(Attack1), P(Attack2)...]
        probs = self.model.predict_proba(feats_scaled)[0]
        
        # In CIC-IDS2017, usually Label 0 is 'Benign' (Normal)
        # We define 'scanning' as anything NOT benign
        p_normal = probs[0]
        p_scanning = 1.0 - p_normal
        
        label = "scanning" if p_scanning >= 0.5 else "normal"
        conf = p_scanning if label == "scanning" else p_normal
        
        return label, conf

# ── Real Random Forest Implementation ──────────────────────────────────────
class RandomForestDetector(BaseDetector):
    def __init__(self):
        super().__init__("rf")

    def predict(self, features):
        if self.model is None:
            return ("normal", 0.5)
            
        feats_reshaped = features.reshape(1, -1)
        feats_scaled = self.scaler.transform(feats_reshaped)
        
        probs = self.model.predict_proba(feats_scaled)[0]
        p_normal = probs[0]
        p_scanning = 1.0 - p_normal
        
        label = "scanning" if p_scanning >= 0.5 else "normal"
        conf = p_scanning if label == "scanning" else p_normal
        
        return label, conf

# ── 1D-CNN Implementation (Placeholder) ────────────────────────────────────
# Note: Since we haven't trained the CNN in Step 1 yet, 
# we keep this as a Mock so the Ensemble code doesn't crash.
class CNNDetector(BaseDetector):
    def __init__(self):
        self.name = "cnn"
        print("[Ensemble] CNN not trained yet. Using MockCNN.")

    def predict(self, features):
        # Mock logic: look for high 'Flow Duration' (index 1 in most schemas)
        val = features[1] if len(features) > 1 else 0
        conf = 0.6 if val > 1000 else 0.4
        label = "scanning" if conf >= 0.5 else "normal"
        return label, conf

# ── Ensemble Engine ────────────────────────────────────────────────────────
class EnsembleIDS:
    def __init__(self, voting="majority", weights=None, threshold=0.5):
        self.detectors = [
            XGBoostDetector(),
            RandomForestDetector(),
            CNNDetector()
        ]
        self.voting = voting
        self.threshold = threshold
        
        # Default equal weights for weighted voting
        if weights is None:
            self.weights = np.ones(len(self.detectors)) / len(self.detectors)
        else:
            self.weights = weights

    def classify(self, features):
        votes = {}
        confs = {}
        p_scan_list = []

        for det in self.detectors:
            label, conf = det.predict(features)
            votes[det.name] = label
            confs[det.name] = conf
            # Probability of scanning for the weighted rule
            p_scan_list.append(conf if label == "scanning" else 1.0 - conf)

        n_scanning = sum(1 for v in votes.values() if v == "scanning")
        
        # Majority Rule: 2 out of 3
        if self.voting == "majority":
            final_label = "scanning" if n_scanning >= 2 else "normal"
        
        # Unanimous Rule: 3 out of 3
        elif self.voting == "unanimous":
            final_label = "scanning" if n_scanning == 3 else "normal"
            
        # Weighted Confidence Rule
        else:
            score = np.dot(self.weights, p_scan_list)
            final_label = "scanning" if score >= self.threshold else "normal"

        return {
            "label": final_label,
            "votes": votes,
            "rule": self.voting
        }