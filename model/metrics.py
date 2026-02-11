"""
Metrics calculation for safety filter model evaluation.
"""

from typing import List, Dict
import numpy as np
import time
import psutil
import os


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
            average_precision_score
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
    def measure_inference_latency(model, X: List[str],
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
    def measure_throughput(model, X: List[str],
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
    def measure_peak_memory(model, X: List[str]) -> float:
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
    def evaluate_model(model, X_test: List[str], y_test: np.ndarray,
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
