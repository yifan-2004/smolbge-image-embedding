"""Retrieval metrics with arbitrary caption counts and image-cluster uncertainty.

Candidate order is the deterministic tie breaker. Main summaries give every
image equal weight; ``text_to_image_query_micro`` additionally reports the
usual query-weighted metrics. With five captions per image these agree.
"""
from __future__ import annotations

import numpy as np


METRIC_NAMES = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")


def _normalized(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError(f"{name} must have nonempty shape (items, dimensions)")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numbers")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    # Rescaling before taking the norm avoids underflow/overflow on valid inputs.
    scale = np.abs(values).max(axis=1, keepdims=True)
    if (scale == 0).any():
        raise ValueError(f"{name} contains a zero vector")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Top k in descending score order; boundary ties prefer smaller indices."""
    k = min(k, len(scores))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[: k - len(above)]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def _rank_metrics(ranks: np.ndarray, ndcg: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "recall_at_1": (ranks <= 1).astype(np.float64),
        "recall_at_5": (ranks <= 5).astype(np.float64),
        "recall_at_10": (ranks <= 10).astype(np.float64),
        "mrr": 1.0 / ranks,
        "ndcg_at_10": ndcg,
    }


def retrieval_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    text_owner: np.ndarray,
    *,
    block_size: int = 256,
) -> dict:
    """Evaluate exact cosine retrieval without assuming five captions per image.

    ``text_owner[j]`` is the image-row index associated with text row ``j``.
    Every image must have at least one caption. Image-to-text relevance is
    binary for every owned caption; text-to-image has one relevant image.
    Memory for similarities is bounded by ``block_size * max(N, M)``.

    ``per_image[direction][metric]`` returns N values in image-row order and is
    directly usable by ``paired_cluster_bootstrap``. Text-to-image values first
    average over the captions of each image, preserving their dependence.
    These arrays are deliberately NumPy arrays, not JSON serializable lists.
    """
    image = _normalized(image_embeddings, "image_embeddings")
    text = _normalized(text_embeddings, "text_embeddings")
    if image.shape[1] != text.shape[1]:
        raise ValueError("image and text embedding dimensions must agree")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    owners = np.asarray(text_owner)
    if owners.shape != (len(text),) or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("text_owner must be an integer vector with one entry per text")
    if (owners < 0).any() or (owners >= len(image)).any():
        raise ValueError("text_owner entries must index the supplied images")
    owners = owners.astype(np.int64, copy=False)
    counts = np.bincount(owners, minlength=len(image))
    if (counts == 0).any():
        raise ValueError("every image must have at least one owned caption")

    discounts = 1.0 / np.log2(np.arange(2, 12))
    image_ranks = np.empty(len(image), dtype=np.int64)
    image_ndcg = np.empty(len(image), dtype=np.float64)
    text_indices = np.arange(len(text))
    for start in range(0, len(image), block_size):
        similarities = image[start : start + block_size] @ text.T
        for local, scores in enumerate(similarities):
            index = start + local
            positives = np.flatnonzero(owners == index)
            best = positives[np.argmax(scores[positives])]
            image_ranks[index] = 1 + np.count_nonzero(scores > scores[best]) + np.count_nonzero(
                (scores == scores[best]) & (text_indices < best)
            )
            top = _top_indices(scores, 10)
            dcg = np.sum((owners[top] == index) * discounts[: len(top)])
            image_ndcg[index] = dcg / discounts[: min(counts[index], 10)].sum()

    text_ranks = np.empty(len(text), dtype=np.int64)
    image_indices = np.arange(len(image))
    for start in range(0, len(text), block_size):
        similarities = text[start : start + block_size] @ image.T
        targets = owners[start : start + len(similarities)]
        positive = similarities[np.arange(len(similarities)), targets, None]
        text_ranks[start : start + len(similarities)] = (
            1
            + (similarities > positive).sum(axis=1)
            + ((similarities == positive) & (image_indices[None, :] < targets[:, None])).sum(axis=1)
        )

    image_values = _rank_metrics(image_ranks, image_ndcg)
    text_values = _rank_metrics(
        text_ranks, np.where(text_ranks <= 10, 1.0 / np.log2(text_ranks + 1), 0.0)
    )
    text_per_image = {
        key: np.bincount(owners, weights=values, minlength=len(image)) / counts
        for key, values in text_values.items()
    }
    return {
        "image_to_text": {key: float(values.mean()) for key, values in image_values.items()},
        "text_to_image": {key: float(values.mean()) for key, values in text_per_image.items()},
        "text_to_image_query_micro": {key: float(values.mean()) for key, values in text_values.items()},
        "per_image": {"image_to_text": image_values, "text_to_image": text_per_image},
        "protocol": {
            "images": len(image),
            "captions": len(text),
            "aggregation": "image_macro",
            "similarity": "cosine",
            "ties": "ascending_candidate_index",
            "ndcg_relevance": "binary_owned_captions_or_paired_image",
        },
    }


def paired_cluster_bootstrap(
    candidate_per_image: np.ndarray,
    baseline_per_image: np.ndarray,
    *,
    seed: int = 42,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """Paired percentile CI over image clusters, conditional on a fixed corpus.

    Supply matching image IDs in exactly the same order. This quantifies query
    sampling uncertainty, not training-seed variation or corpus resampling.
    Differences are in score units (multiply by 100 for percentage points).
    """
    candidate = np.asarray(candidate_per_image, dtype=np.float64)
    baseline = np.asarray(baseline_per_image, dtype=np.float64)
    if candidate.ndim != 1 or len(candidate) < 2 or candidate.shape != baseline.shape:
        raise ValueError("paired scores must be matching one-dimensional arrays with at least two images")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("paired scores must be finite")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    # Avoid a resamples-by-dataset-size allocation for larger evaluations.
    for index in range(n_resamples):
        draws[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_images": len(differences),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "seed": int(seed),
        "unit": "image_cluster",
    }
