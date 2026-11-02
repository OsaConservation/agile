"""Saved Hoplite classifier loading and exact, batched boundary selection."""

import base64
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SavedClassifier:
    classes: tuple[str, ...]
    model_config: dict
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def load(cls, path: Path):
        return cls.from_bytes(path.read_bytes())

    @classmethod
    def from_bytes(cls, content: bytes):
        """Decode the same bytes that will be retained as run provenance."""
        config = json.loads(content)
        classes = tuple(config['classes'])
        if (not classes or any(not isinstance(label, str) or not label for label in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError('Classifier must have distinct, nonempty classes')
        weights = np.frombuffer(base64.b64decode(config['beta'], validate=True), dtype=np.float32)
        biases = np.frombuffer(base64.b64decode(config['beta_bias'], validate=True), dtype=np.float32)
        dimension = config['model_config']['embedding_dim']
        if type(dimension) is not int or dimension < 1:
            raise ValueError('Classifier embedding dimension must be a positive integer')
        if weights.size != dimension * len(classes) or biases.size != len(classes):
            raise ValueError('Classifier dimensions do not match its metadata')
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError('Classifier contains nonfinite parameters')
        return cls(classes, config['model_config'], weights.reshape(dimension, len(classes)), biases)


def select_candidates(db, window_ids, classifier, count=15, batch_size=1024):
    """Return species -> IDs, ordered by (absolute signed logit, source ID).

    Every supplied window is scored. Only one batch plus count candidates per
    species is retained. No probability threshold or approximate search is used.
    """
    if count < 1 or batch_size < 1:
        raise ValueError('count and batch_size must be positive')

    best = {species: [] for species in classifier.classes}
    for start in range(0, len(window_ids), batch_size):
        ids = window_ids[start:start + batch_size]
        embeddings = np.asarray(db.get_embeddings_batch(ids))
        if embeddings.shape != (len(ids), classifier.weights.shape[0]):
            raise ValueError('Embedding dimensions do not match classifier')

        for column, species in enumerate(classifier.classes):
            # Exporter recomputes each dot product; use identical scalar scoring
            # so roundoff cannot change membership at a close boundary tie.
            scores = [
                (abs(float(np.dot(embedding, classifier.weights[:, column])
                           + classifier.biases[column])), int(wid))
                for wid, embedding in zip(ids, embeddings)
            ]
            if not all(np.isfinite(distance) for distance, _ in scores):
                raise ValueError('Classifier produced nonfinite logits')
            best[species] = sorted(best[species] + scores)[:count]

    return {species: [wid for _, wid in values] for species, values in best.items()}
