"""
ensemble.py — Multi-model ensemble IDS for adversarial RL thesis.

This module implements the core defensive layer. It integrates three 
structurally diverse detectors: XGBoost (Tree-based), 1D-CNN (Neural), 
and Random Forest (Bagged Ensemble). 

It supports three distinct aggregation rules (Majority, Unanimous, Weighted) 
to evaluate how collective decision-making impacts adversarial resilience.
"""

import json
import joblib
import numpy as np
import os
import pandas as pd
import warnings
from abc import ABC, abstractmethod

# Suppress technical warnings for a cleaner console during long training
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Paths for the Mac Pro filesystem
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
STANDALONE_RULES = ("xgboost", "rf", "cnn")
VOTING_RULES = ("majority", "unanimous", "weighted", "any") + STANDALONE_RULES

# ── Detector Interface ─────────────────────────────────────────────────────
class BaseDetector(ABC):
    """
    Abstract interface for all constituent detectors. 
    Ensures that real models and mock models can be swapped interchangeably.
    """
    def __init__(self, model_name):
        self.name = model_name
        self.model = None
        self.scaler = None
        self.is_mock = True
        self.load()

    def load(self):
        """Loads trained artifacts from the /models directory."""
        model_path = os.path.join(MODEL_DIR, f"{self.name}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_mock = False
                # One-row inference: thread start-up costs more than it saves
                if hasattr(self.model, "n_jobs"):
                    self.model.set_params(n_jobs=1)
                print(f"[Ensemble] STATUS: Real {self.name} model loaded successfully.")
            except Exception as e:
                print(f"[Ensemble] ERROR: Failed to load {self.name}: {e}")
        else:
            print(f"[Ensemble] WARNING: {self.name} artifacts missing. Falling back to MOCK.")

    def scale_features(self, features):
        """Prepares raw features for model inference using the trained scaler."""
        reshaped = features.reshape(1, -1)
        if hasattr(self.scaler, "feature_names_in_"):
            reshaped = pd.DataFrame(reshaped, columns=self.scaler.feature_names_in_)
        return self.scaler.transform(reshaped)

    @abstractmethod
    def predict(self, features):
        pass

# ── Constituent Models ──────────────────────────────────────────────────────

class XGBoostDetector(BaseDetector):
    """XGBoost provides sharp decision boundaries for tabular data."""
    def __init__(self):
        super().__init__("xgboost")

    def predict(self, features):
        if self.is_mock:
            # Original Mock Logic (Kept for thesis comparison)
            aggr = features[0] if len(features) > 0 else 0
            conf = 0.8 if aggr > 0.7 else 0.2
            label = "scanning" if conf >= 0.5 else "normal"
            return label, conf
        
        # Real Inference
        scaled = self.scale_features(features)
        probs = self.model.predict_proba(scaled)[0]
        p_scanning = 1.0 - probs[0] # Assuming Index 0 is 'Benign'
        label = "scanning" if p_scanning >= 0.5 else "normal"
        conf = p_scanning if label == "scanning" else probs[0]
        return label, conf

class RandomForestDetector(BaseDetector):
    """Random Forest offers variance reduction through bagging."""
    def __init__(self):
        super().__init__("rf")

    def predict(self, features):
        if self.is_mock:
            syn = features[1] if len(features) > 1 else 0
            conf = 0.7 if syn > 0.5 else 0.3
            label = "scanning" if conf >= 0.5 else "normal"
            return label, conf
            
        scaled = self.scale_features(features)
        probs = self.model.predict_proba(scaled)[0]
        p_scanning = 1.0 - probs[0]
        label = "scanning" if p_scanning >= 0.5 else "normal"
        conf = p_scanning if label == "scanning" else probs[0]
        return label, conf

class CNNDetector(BaseDetector):
    """
    1D-CNN that captures sequential patterns in traffic feature vectors.

    Overrides BaseDetector.load() to handle PyTorch .pth weights instead
    of a joblib .pkl file. Falls back to mock behaviour if the trained
    artifacts are not yet present (i.e. train_cnn.py hasn't been run).
    """

    def __init__(self):
        # _device must exist before super().__init__() calls self.load()
        self._device = "cpu"
        super().__init__("cnn")

    # ── Override: PyTorch load instead of joblib ───────────────────────────
    def load(self):
        arch_path   = os.path.join(MODEL_DIR, "cnn_arch.json")
        model_path  = os.path.join(MODEL_DIR, "cnn_model.pth")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

        if all(os.path.exists(p) for p in [arch_path, model_path, scaler_path]):
            try:
                import torch
                from cnn_arch import IDS1DCNN

                with open(arch_path) as f:
                    arch = json.load(f)

                net = IDS1DCNN(**arch)
                net.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
                net.eval()

                self.model  = net
                self.scaler = joblib.load(scaler_path)
                self.is_mock = False
                print(f"[Ensemble] STATUS: Real CNN model loaded successfully.")
            except Exception as e:
                print(f"[Ensemble] ERROR: Failed to load CNN: {e}")
        else:
            missing = [os.path.basename(p) for p in [arch_path, model_path, scaler_path]
                       if not os.path.exists(p)]
            print(
                f"[Ensemble] WARNING: CNN artifacts missing {missing}. "
                f"Run train_cnn.py first. Falling back to MOCK."
            )

    def predict(self, features):
        if self.is_mock:
            # Fallback mock: kept identical to original for thesis comparison
            val  = features[1] if len(features) > 1 else 0
            conf = 0.6 if val > 1000 else 0.4
            label = "scanning" if conf >= 0.5 else "normal"
            return label, conf

        # ── Real PyTorch inference ─────────────────────────────────────────
        import torch
        scaled = self.scale_features(features)               # (1, n_features)
        x = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)                           # (1, n_classes)
            probs  = torch.softmax(logits, dim=1)[0].numpy()

        # Index 0 = BENIGN (LabelEncoder sorts alphabetically)
        p_scanning = 1.0 - probs[0]
        label = "scanning" if p_scanning >= 0.5 else "normal"
        conf  = p_scanning if label == "scanning" else float(probs[0])
        return label, conf

# ── Aggregation Engine ─────────────────────────────────────────────────────

def load_weights():
    """
    Confidence-weighted rule: per-detector weights from held-out performance,
    written by evaluate_detectors.py. Equal weights if not yet computed.
    """
    path = os.path.join(MODEL_DIR, "ensemble_weights.json")
    if os.path.exists(path):
        with open(path) as f:
            w = json.load(f)
        return np.array([w["xgboost"], w["rf"], w["cnn"]], dtype=np.float64)
    return np.ones(3) / 3


class EnsembleIDS:
    def __init__(self, voting="majority", weights=None, threshold=0.5, detectors=None):
        if voting not in VOTING_RULES:
            raise ValueError(f"Unknown voting rule {voting!r}; choose from {VOTING_RULES}")
        # Pass `detectors` to share already-loaded models between rule instances
        self.detectors = detectors or [XGBoostDetector(), RandomForestDetector(), CNNDetector()]
        self.voting = voting
        self.threshold = threshold
        self.weights = np.asarray(weights) if weights is not None else load_weights()
        # Standalone rules run a single detector (RQ1 baseline)
        self._active = [d for d in self.detectors if d.name == voting] or self.detectors

    def classify(self, features):
        """Performs collective inference based on selected voting rule."""
        votes, confs = {}, {}
        for det in self._active:
            votes[det.name], confs[det.name] = det.predict(features)
        return self.decide(votes, confs)

    def decide(self, votes, confs):
        """
        Applies the voting rule to per-detector (label, confidence) outputs.
        Separate from classify() so one set of predictions can be scored
        under every rule (paired comparison in evaluate_agents.py).
        """
        names = [d.name for d in self._active]
        votes = {n: votes[n] for n in names}
        confs = {n: confs[n] for n in names}
        p_scan_list = [confs[n] if votes[n] == "scanning" else 1.0 - confs[n] for n in names]
        n_scanning = sum(1 for v in votes.values() if v == "scanning")

        if self.voting in STANDALONE_RULES:
            final_label = votes[self.voting]
            score = p_scan_list[0]
        elif self.voting == "majority":
            final_label = "scanning" if n_scanning >= 2 else "normal"
            score = n_scanning / 3.0
        elif self.voting == "unanimous":
            # All three must flag: the most permissive rule for the attacker
            final_label = "scanning" if n_scanning == 3 else "normal"
            score = n_scanning / 3.0
        elif self.voting == "any":
            # Any single detector flags: the most defensive rule
            final_label = "scanning" if n_scanning >= 1 else "normal"
            score = n_scanning / 3.0
        else: # Weighted
            score = float(np.dot(self.weights, p_scan_list))
            final_label = "scanning" if score >= self.threshold else "normal"

        return {
            "label": final_label,
            "votes": votes,
            "confidences": confs,
            "score": score,
            "rule": self.voting
        }