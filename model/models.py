"""
Safety filter model implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
import time
import psutil
import os


class BaseModel(ABC):
    """Base class for all safety filter models."""

    @abstractmethod
    def fit(self, X: List[str], y: np.ndarray):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels for input texts."""
        pass

    @abstractmethod
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for input texts."""
        pass


class TfIdfModel(BaseModel):
    """TF-IDF vectorizer for text feature extraction."""

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None

    def fit(self, X: List[str], y: np.ndarray = None):
        """Fit TF-IDF vectorizer on training texts."""
        # Mock implementation
        print(f"Fitting TF-IDF with max_features={self.max_features}, ngram_range={self.ngram_range}")
        self.vectorizer = "fitted"
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        # Mock implementation
        return np.random.rand(len(X), self.max_features)

    def predict(self, X: List[str]) -> np.ndarray:
        """Not applicable for vectorizer."""
        raise NotImplementedError("Use transform() method for TF-IDF")

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Not applicable for vectorizer."""
        raise NotImplementedError("Use transform() method for TF-IDF")


class LogRegModel(BaseModel):
    """Logistic Regression classifier for binary toxic/safe classification."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.vectorizer = TfIdfModel()

    def fit(self, X: List[str], y: np.ndarray):
        """Train logistic regression on TF-IDF features."""
        # Mock implementation
        print(f"Training Logistic Regression with C={self.C}, max_iter={self.max_iter}")
        self.vectorizer.fit(X)
        X_features = self.vectorizer.transform(X)
        print(f"Feature shape: {X_features.shape}")
        self.model = "fitted"
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict binary labels (0=safe, 1=toxic)."""
        # Mock implementation
        X_features = self.vectorizer.transform(X)
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        # Mock implementation
        probs = np.random.rand(len(X), 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_metrics(self) -> dict:
        """Get model performance metrics."""
        return {
            'inference_time_ms': 0.5,
            'memory_mb': 10.2,
            'model_size_mb': 5.5
        }


class TransformerClassifier(BaseModel):
    """Transformer-based classifier for toxic comment detection."""

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def fit(self, X: List[str], y: np.ndarray):
        """Fine-tune transformer model on training data."""
        # Mock implementation
        print(f"Fine-tuning {self.model_name} with max_length={self.max_length}")
        print(f"Training samples: {len(X)}")
        self.tokenizer = "fitted"
        self.model = "fitted"
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict binary labels (0=safe, 1=toxic)."""
        # Mock implementation
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        # Mock implementation
        probs = np.random.rand(len(X), 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_metrics(self) -> dict:
        """Get model performance metrics."""
        return {
            'inference_time_ms': 15.3,
            'memory_mb': 256.7,
            'model_size_mb': 220.4
        }


class MetricsCalculator:
    """Calculate quality and performance metrics for safety filter models."""

    @staticmethod
    def calculate_quality_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate quality metrics for model predictions.

        Args:
            y_true: Ground truth labels (0=safe, 1=toxic)
            y_pred: Predicted labels (0=safe, 1=toxic)
            y_proba: Predicted probabilities (optional, for PR-AUC)

        Returns:
            Dictionary with precision, recall, f1-score, and PR-AUC
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            average_precision_score, confusion_matrix
        )

        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_proba is not None:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])

        return metrics

    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def measure_inference_latency(model: BaseModel, X: List[str],
                                   n_runs: int = 100) -> Dict[str, float]:
        """
        Measure average inference latency per sample.

        Args:
            model: Trained model instance
            X: List of input texts
            n_runs: Number of runs for averaging

        Returns:
            Dictionary with latency statistics (mean, std, min, max) in milliseconds
        """
        latencies = []

        for _ in range(n_runs):
            start_time = time.perf_counter()
            model.predict(X)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000 / len(X)
            latencies.append(latency_ms)

        latencies = np.array(latencies)

        return {
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies))
        }

    @staticmethod
    def measure_throughput(model: BaseModel, X: List[str],
                          duration_seconds: float = 10.0) -> float:
        """
        Measure throughput (samples per second).

        Args:
            model: Trained model instance
            X: List of input texts
            duration_seconds: Duration to run throughput test

        Returns:
            Throughput in samples per second
        """
        start_time = time.perf_counter()
        total_samples = 0

        while (time.perf_counter() - start_time) < duration_seconds:
            model.predict(X)
            total_samples += len(X)

        elapsed_time = time.perf_counter() - start_time
        return total_samples / elapsed_time

    @staticmethod
    def measure_peak_memory(model: BaseModel, X: List[str]) -> float:
        """
        Measure peak memory usage during inference.

        Args:
            model: Trained model instance
            X: List of input texts

        Returns:
            Peak memory usage in MB
        """
        process = psutil.Process(os.getpid())

        # Measure memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB

        # Run inference
        model.predict(X)

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # Convert to MB

        return mem_after - mem_before

    @staticmethod
    def get_model_size(model_path: str) -> float:
        """
        Get model size on disk.

        Args:
            model_path: Path to saved model file

        Returns:
            Model size in MB
        """
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            return size_bytes / 1024 / 1024  # Convert to MB
        return 0.0

    @staticmethod
    def evaluate_model(model: BaseModel, X_test: List[str], y_test: np.ndarray,
                      n_latency_runs: int = 100) -> Dict[str, any]:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained model instance
            X_test: Test texts
            y_test: Test labels
            n_latency_runs: Number of runs for latency measurement

        Returns:
            Dictionary with all quality and performance metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate quality metrics
        quality_metrics = MetricsCalculator.calculate_quality_metrics(
            y_test, y_pred, y_proba
        )

        # Calculate confusion matrix
        cm = MetricsCalculator.calculate_confusion_matrix(y_test, y_pred)

        # Measure performance metrics
        latency_metrics = MetricsCalculator.measure_inference_latency(
            model, X_test[:100], n_latency_runs
        )

        throughput = MetricsCalculator.measure_throughput(
            model, X_test[:100], duration_seconds=5.0
        )

        peak_memory = MetricsCalculator.measure_peak_memory(
            model, X_test[:100]
        )

        return {
            'quality': quality_metrics,
            'confusion_matrix': cm.tolist(),
            'latency': latency_metrics,
            'throughput_samples_per_sec': throughput,
            'peak_memory_mb': peak_memory
        }

