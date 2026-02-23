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
            random_state=42
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


class LoRATransformerClassifier(BaseModel):
    """
    LoRA-adapted Transformer classifier for binary toxic/safe classification.
    Uses PEFT LoRA adapters on a HuggingFace pretrained model to reduce
    trainable parameters while maintaining quality.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128,
                 model_dir: Optional[str] = None, batch_size: int = 64,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 target_modules: Optional[List[str]] = None):
        import torch

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_lin", "v_lin"]
        self.model = None
        self.tokenizer = None

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if model_dir:
            self.load(model_dir)

    def load(self, model_dir: str):
        """Load merged LoRA model and tokenizer from directory."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"LoRA-merged model loaded from {model_dir} (device: {self.device})")

    def save(self, model_dir: str):
        """Merge LoRA weights and save full model to directory."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model to save")

        from peft import PeftModel

        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, PeftModel):
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(model_dir)
        else:
            self.model.save_pretrained(model_dir)

        self.tokenizer.save_pretrained(model_dir)
        print(f"LoRA-merged model saved to {model_dir}")

    def fit(self, X: List[str], y: np.ndarray, epochs: int = 3,
            batch_size: int = 32, lr: float = 2e-5, warmup_ratio: float = 0.1,
            max_grad_norm: float = 1.0):
        """Fine-tune transformer with LoRA adapters on training data."""
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        from peft import get_peft_model, LoraConfig, TaskType
        from tqdm.auto import tqdm

        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()

        class _TextDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts if isinstance(texts, list) else texts.tolist()
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx], int(self.labels[idx])

        tokenizer_ref = self.tokenizer
        max_len = self.max_length

        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer_ref(
                list(texts), truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )
            encodings['labels'] = torch.tensor(labels, dtype=torch.long)
            return encodings

        dataset = _TextDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn)

        total_steps = len(loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} done  avg_loss={avg_loss:.4f}")

        self.model.eval()
        return self

    def _predict_batched(self, X: List[str]):
        """Run batched inference, return raw logits."""
        import torch

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded")

        all_logits = []
        for i in range(0, len(X), self.batch_size):
            batch_texts = X[i:i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt"
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
            all_logits.append(outputs.logits.cpu())

        return torch.cat(all_logits, dim=0)

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict binary labels (0=safe, 1=toxic)."""
        logits = self._predict_batched(X)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        import torch
        logits = self._predict_batched(X)
        return torch.softmax(logits, dim=1).numpy()

    def get_metrics(self, X_test: List[str], y_test: np.ndarray,
                   n_latency_runs: int = 100) -> dict:
        from .metrics import MetricsCalculator
        return MetricsCalculator.evaluate_model(
            self, X_test, y_test, n_latency_runs
        )


class TransformerClassifier(BaseModel):
    """
    Lightweight Transformer-based classifier for binary toxic/safe classification.
    Uses HuggingFace pretrained models with a sequence classification head.
    Can load fine-tuned models or train from scratch.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128,
                 model_dir: Optional[str] = None, batch_size: int = 64):
        """
        Initialize model.

        Args:
            model_name: HuggingFace model identifier for pretrained weights.
            max_length: Maximum token sequence length.
            model_dir: Path to saved fine-tuned model directory (optional).
            batch_size: Batch size for inference.
        """
        import torch

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if model_dir:
            self.load(model_dir)

    def load(self, model_dir: str):
        """Load fine-tuned model and tokenizer from directory."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"Transformer model loaded from {model_dir} (device: {self.device})")

    def save(self, model_dir: str):
        """Save fine-tuned model and tokenizer to directory."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model to save")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        print(f"Transformer model saved to {model_dir}")

    def fit(self, X: List[str], y: np.ndarray, epochs: int = 3,
            batch_size: int = 32, lr: float = 2e-5, warmup_ratio: float = 0.1,
            max_grad_norm: float = 1.0):
        """Fine-tune transformer on training data."""
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        from tqdm.auto import tqdm

        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.to(self.device)

        class _TextDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts if isinstance(texts, list) else texts.tolist()
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx], int(self.labels[idx])

        tokenizer_ref = self.tokenizer
        max_len = self.max_length

        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer_ref(
                list(texts), truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )
            encodings['labels'] = torch.tensor(labels, dtype=torch.long)
            return encodings

        dataset = _TextDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn)

        total_steps = len(loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} done  avg_loss={avg_loss:.4f}")

        self.model.eval()
        return self

    def _predict_batched(self, X: List[str]):
        """Run batched inference, return raw logits."""
        import torch

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded")

        all_logits = []
        for i in range(0, len(X), self.batch_size):
            batch_texts = X[i:i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt"
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
            all_logits.append(outputs.logits.cpu())

        return torch.cat(all_logits, dim=0)

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict binary labels (0=safe, 1=toxic)."""
        logits = self._predict_batched(X)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for each class."""
        import torch
        logits = self._predict_batched(X)
        return torch.softmax(logits, dim=1).numpy()

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


