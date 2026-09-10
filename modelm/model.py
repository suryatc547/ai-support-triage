"""Custom lightweight neural classifier for domain ticket classification.

Built in pure NumPy to run on CPU with zero dependencies beyond numpy,
consuming < 15 MB RAM and executing in < 5 milliseconds.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _tokenize(text: str) -> list[str]:
    """Extract normalized word tokens and character 3-4-grams for subword matching."""
    text = (text or "").lower()
    words = re.findall(r"\b[a-z0-9_-]+\b", text)

    features = list(words)
    # Add word bigrams for context (e.g., 'vpn connection', 'invoice approval')
    for i in range(len(words) - 1):
        features.append(f"{words[i]}_{words[i + 1]}")

    # Add character n-grams for words >= 4 chars to handle plurals, typos, and affixes
    for w in words:
        if len(w) >= 4:
            for n in (3, 4):
                for i in range(len(w) - n + 1):
                    features.append(f"#{w[i : i + n]}")

    return features


class CustomTicketClassifier:
    """A lightweight Softmax classifier with TF-IDF representation and AdamW optimization."""

    def __init__(
        self,
        categories: Sequence[str] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.categories: list[str] = list(categories or [])
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.weights: np.ndarray | None = None  # shape: (vocab_size, num_classes)
        self.bias: np.ndarray | None = None  # shape: (num_classes,)
        self.temperature: float = max(temperature, 1e-3)

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[str] | None = None,
        epochs: int = 250,
        lr: float = 0.05,
        optimizer: str = "adamw",
        weight_decay: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        warmup_ratio: float = 0.1,
        reg: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Train the model with AdamW or GD with cross-entropy loss and validation tracking."""
        # 1. Collect unique categories
        if not self.categories:
            self.categories = sorted(set(labels))
        num_classes = len(self.categories)
        cat_to_idx = {c: i for i, c in enumerate(self.categories)}

        # 2. Build vocabulary and compute document frequencies
        tokenized_corpus = [_tokenize(t) for t in texts]
        df_counts: dict[str, int] = {}
        for tokens in tokenized_corpus:
            for token in set(tokens):
                df_counts[token] = df_counts.get(token, 0) + 1

        n_docs = len(texts)
        vocab_list = sorted([token for token, count in df_counts.items() if count >= 1])
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        vocab_size = len(self.vocab)

        # 3. Compute smooth IDF
        self.idf = [
            math.log((n_docs + 1.0) / (df_counts[token] + 1.0)) + 1.0 for token in vocab_list
        ]
        idf_arr = np.array(self.idf, dtype=np.float32)

        # 4. Vectorize training documents into TF-IDF matrix
        x_mat = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(tokenized_corpus):
            for token in tokens:
                idx = self.vocab.get(token)
                if idx is not None:
                    x_mat[i, idx] += 1.0
            x_mat[i] = x_mat[i] * idf_arr
            norm = np.linalg.norm(x_mat[i])
            if norm > 0:
                x_mat[i] = x_mat[i] / norm

        y_indices = np.array([cat_to_idx[lbl] for lbl in labels], dtype=np.int32)
        y_onehot = np.zeros((n_docs, num_classes), dtype=np.float32)
        y_onehot[np.arange(n_docs), y_indices] = 1.0

        # Vectorize validation documents if provided
        has_val = val_texts is not None and val_labels is not None and len(val_texts) > 0
        if has_val:
            val_indices = np.array([cat_to_idx.get(lbl, 0) for lbl in val_labels], dtype=np.int32)
            val_mat = np.zeros((len(val_texts), vocab_size), dtype=np.float32)
            for i, vtext in enumerate(val_texts):
                v_tokens = _tokenize(vtext)
                for token in v_tokens:
                    idx = self.vocab.get(token)
                    if idx is not None:
                        val_mat[i, idx] += 1.0
                val_mat[i] = val_mat[i] * idf_arr
                norm = np.linalg.norm(val_mat[i])
                if norm > 0:
                    val_mat[i] = val_mat[i] / norm
        else:
            val_indices = np.array([], dtype=np.int32)
            val_mat = np.zeros((0, vocab_size), dtype=np.float32)

        # 5. Initialize weights and optimizer states
        np.random.seed(42)
        self.weights = np.random.randn(vocab_size, num_classes).astype(np.float32) * 0.01
        self.bias = np.zeros(num_classes, dtype=np.float32)

        m_w = np.zeros_like(self.weights)
        v_w = np.zeros_like(self.weights)
        m_b = np.zeros_like(self.bias)
        v_b = np.zeros_like(self.bias)

        # Auto-detect legacy call (e.g. lr=2.0) vs modern AdamW
        opt_mode = "gd" if (lr >= 1.0 and optimizer == "adamw") else optimizer.lower()

        warmup_epochs = max(1, int(epochs * warmup_ratio))
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()

        # 6. Training loop
        for ep in range(1, epochs + 1):
            logits = np.dot(x_mat, self.weights) + self.bias
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(shift_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            train_loss = float(
                -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
            )
            history["train_loss"].append(train_loss)

            grad_logits = (probs - y_onehot) / n_docs
            grad_w = np.dot(x_mat.T, grad_logits)
            grad_b = np.sum(grad_logits, axis=0)

            if opt_mode == "adamw":
                # Learning rate warmup + cosine annealing schedule
                if ep <= warmup_epochs:
                    lr_t = lr * (ep / warmup_epochs)
                else:
                    progress = (ep - warmup_epochs) / max(1, (epochs - warmup_epochs))
                    lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * progress))

                m_w = beta1 * m_w + (1.0 - beta1) * grad_w
                v_w = beta2 * v_w + (1.0 - beta2) * (grad_w**2)
                m_w_hat = m_w / (1.0 - (beta1**ep))
                v_w_hat = v_w / (1.0 - (beta2**ep))
                self.weights -= lr_t * (
                    m_w_hat / (np.sqrt(v_w_hat) + eps) + weight_decay * self.weights
                )

                m_b = beta1 * m_b + (1.0 - beta1) * grad_b
                v_b = beta2 * v_b + (1.0 - beta2) * (grad_b**2)
                m_b_hat = m_b / (1.0 - (beta1**ep))
                v_b_hat = v_b / (1.0 - (beta2**ep))
                self.bias -= lr_t * (m_b_hat / (np.sqrt(v_b_hat) + eps))
            else:
                # Vanilla Gradient Descent fallback
                self.weights -= lr * (grad_w + reg * self.weights)
                self.bias -= lr * grad_b

            # Validation checkpointing
            if has_val:
                val_logits = np.dot(val_mat, self.weights) + self.bias
                val_shift = val_logits - np.max(val_logits, axis=1, keepdims=True)
                val_exp = np.exp(val_shift)
                val_probs = val_exp / np.sum(val_exp, axis=1, keepdims=True)

                val_onehot = np.zeros((len(val_texts), num_classes), dtype=np.float32)
                val_onehot[np.arange(len(val_texts)), val_indices] = 1.0
                v_loss = float(
                    -np.mean(np.sum(val_onehot * np.log(np.clip(val_probs, 1e-12, 1.0)), axis=1))
                )
                v_preds = np.argmax(val_logits, axis=1)
                v_acc = float(np.mean(v_preds == val_indices))

                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)

                if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss < best_val_loss):
                    best_val_acc = v_acc
                    best_val_loss = v_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias.copy()

        # Restore best checkpoint if validation set was used
        if has_val and best_val_acc >= 0:
            self.weights = best_weights
            self.bias = best_bias

        return history

    def _vectorize(self, text: str) -> np.ndarray:
        """Convert an input text into an L2-normalized TF-IDF vector."""
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tokens = _tokenize(text)
        for t in tokens:
            idx = self.vocab.get(t)
            if idx is not None:
                vec[idx] += 1.0
        vec = vec * np.array(self.idf, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def predict_proba(self, text: str, temperature: float | None = None) -> dict[str, float]:
        """Compute class probabilities with optional temperature scaling."""
        if self.weights is None or self.bias is None or not self.vocab:
            raise RuntimeError("Model has not been trained or loaded.")

        temp = temperature if temperature is not None else self.temperature
        temp = max(temp, 1e-3)
        vec = self._vectorize(text)
        logits = (np.dot(vec, self.weights) + self.bias) / temp
        shift_logits = logits - np.max(logits)
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / np.sum(exp_logits)
        return {cat: float(probs[i]) for i, cat in enumerate(self.categories)}

    def predict(self, text: str, temperature: float | None = None) -> tuple[str, float]:
        """Return the best predicted category and its confidence score."""
        probs = self.predict_proba(text, temperature=temperature)
        best_cat = max(probs, key=lambda k: probs[k])
        return best_cat, probs[best_cat]

    def save(self, filepath: str | Path) -> None:
        """Save model parameters and vocabulary to a JSON file."""
        data = {
            "categories": self.categories,
            "vocab": self.vocab,
            "idf": self.idf,
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": self.bias.tolist() if self.bias is not None else [],
            "temperature": self.temperature,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str | Path) -> CustomTicketClassifier:
        """Load model parameters and vocabulary from a JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        model = cls(
            categories=data.get("categories", []),
            temperature=data.get("temperature", 1.0),
        )
        model.vocab = data.get("vocab", {})
        model.idf = data.get("idf", [])
        model.weights = np.array(data.get("weights", []), dtype=np.float32)
        model.bias = np.array(data.get("bias", []), dtype=np.float32)
        return model
