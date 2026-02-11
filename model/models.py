"""
Safety filter model implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pickle
from pathlib import Path


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
    """
    TF-IDF + Logistic Regression classifier for binary toxic/safe classification.
    Can load pre-trained models or train from scratch.
    """

    def __init__(self, vectorizer_path: Optional[str] = None, model_path: Optional[str] = None,
                 C: float = 1.0, max_iter: int = 1000):
        """
        Initialize model.

        Args:
            vectorizer_path: Path to saved TF-IDF vectorizer (optional)
            model_path: Path to saved LogReg model (optional)
            C: Regularization parameter for LogisticRegression
            max_iter: Maximum iterations for LogisticRegression
        """
        self.C = C
        self.max_iter = max_iter
        self.vectorizer = None
        self.model = None

        if vectorizer_path and model_path:
            self.load(vectorizer_path, model_path)

    def load(self, vectorizer_path: str, model_path: str):
        """Load pre-trained model and vectorizer."""
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        print(f"Model loaded from {model_path}")
        print(f"Vectorizer loaded from {vectorizer_path}")

    def fit(self, X: List[str], y: np.ndarray):
        """Train TF-IDF + Logistic Regression model."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        print("Training TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8
        )

        X_tfidf = self.vectorizer.fit_transform(X)

        print(f"Training Logistic Regression with C={self.C}, max_iter={self.max_iter}...")
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_tfidf, y)
        print("Training complete!")
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict binary labels (0=safe, 1=toxic)."""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained or loaded")

        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained or loaded")

        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)

    def save(self, vectorizer_path: str, model_path: str):
        """Save trained model and vectorizer."""
        if self.vectorizer is None or self.model is None:
            raise ValueError("No model to save")

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")

    def get_metrics(self, X_test: List[str], y_test: np.ndarray,
                   n_latency_runs: int = 100) -> dict:
        """
        Calculate comprehensive model metrics using MetricsCalculator.

        Args:
            X_test: Test texts
            y_test: Test labels
            n_latency_runs: Number of runs for latency measurement

        Returns:
            Dictionary with quality and performance metrics
        """
        from .metrics import MetricsCalculator

        return MetricsCalculator.evaluate_model(
            self, X_test, y_test, n_latency_runs
        )


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

    def get_metrics(self, X_test: List[str], y_test: np.ndarray,
                   n_latency_runs: int = 100) -> dict:
        """
        Calculate comprehensive model metrics using MetricsCalculator.

        Args:
            X_test: Test texts
            y_test: Test labels
            n_latency_runs: Number of runs for latency measurement

        Returns:
            Dictionary with quality and performance metrics
        """
        from .metrics import MetricsCalculator

        return MetricsCalculator.evaluate_model(
            self, X_test, y_test, n_latency_runs
        )


