"""Semantic consistency scoring using sentence embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings for a list of texts."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_semantic_consistency(responses: list[str]) -> dict:
    """Compute semantic consistency across a set of responses.

    Measures how similar all responses are to each other using
    pairwise cosine similarity of embeddings.

    Returns:
        dict with:
        - mean_similarity: average pairwise cosine similarity
        - min_similarity: minimum pairwise similarity (worst case)
        - max_similarity: maximum pairwise similarity
        - std_similarity: standard deviation
        - pairwise_scores: list of all pairwise similarities
    """
    if len(responses) < 2:
        return {
            "mean_similarity": 1.0,
            "min_similarity": 1.0,
            "max_similarity": 1.0,
            "std_similarity": 0.0,
            "pairwise_scores": [],
        }

    # Filter out empty/error responses
    valid_responses = [r for r in responses if r and not r.startswith("ERROR:")]
    if len(valid_responses) < 2:
        return {
            "mean_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "std_similarity": 0.0,
            "pairwise_scores": [],
        }

    embeddings = get_embeddings(valid_responses)
    pairwise = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            pairwise.append(sim)

    return {
        "mean_similarity": float(np.mean(pairwise)),
        "min_similarity": float(np.min(pairwise)),
        "max_similarity": float(np.max(pairwise)),
        "std_similarity": float(np.std(pairwise)),
        "pairwise_scores": pairwise,
    }


def compute_similarity_to_reference(responses: list[str], reference: str) -> dict:
    """Compute similarity of each response to a reference answer.

    Returns:
        dict with:
        - scores: list of similarity scores to reference
        - mean: average similarity
        - std: standard deviation
    """
    if not responses or not reference:
        return {"scores": [], "mean": 0.0, "std": 0.0}

    all_texts = [reference] + responses
    embeddings = get_embeddings(all_texts)
    ref_emb = embeddings[0]
    resp_embs = embeddings[1:]

    scores = [cosine_similarity(ref_emb, e) for e in resp_embs]

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
