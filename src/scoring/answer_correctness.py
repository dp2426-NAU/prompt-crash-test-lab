"""Answer correctness scoring for grounded Q&A task."""

import re

import numpy as np

from .semantic_similarity import get_embeddings, cosine_similarity


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def keyword_overlap(response: str, ground_truth: str) -> float:
    """Compute keyword overlap between response and ground truth."""
    resp_words = set(normalize_text(response).split())
    truth_words = set(normalize_text(ground_truth).split())

    if not truth_words:
        return 0.0

    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                  "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
                  "and", "or", "but", "not", "that", "this", "these", "those"}

    resp_keywords = resp_words - stop_words
    truth_keywords = truth_words - stop_words

    if not truth_keywords:
        return 1.0

    overlap = len(resp_keywords & truth_keywords)
    return overlap / len(truth_keywords)


def check_citation_present(response: str, context: str) -> dict:
    """Check if the response includes a citation/quote from the context.

    Returns:
        dict with:
        - has_quote: bool
        - quote_accuracy: float (how much of a found quote matches context)
    """
    # Look for quoted text
    quote_patterns = [
        r'"([^"]{10,})"',
        r"'([^']{10,})'",
        r"Quote:\s*(.+)",
        r"Supporting [Qq]uote:\s*(.+)",
    ]

    for pattern in quote_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            match_normalized = normalize_text(match)
            context_normalized = normalize_text(context)
            if match_normalized in context_normalized:
                return {"has_quote": True, "quote_accuracy": 1.0}
            # Partial match
            overlap = keyword_overlap(match, context)
            if overlap > 0.5:
                return {"has_quote": True, "quote_accuracy": overlap}

    return {"has_quote": False, "quote_accuracy": 0.0}


def compute_answer_correctness(response: str, ground_truth: dict, context: str = "") -> dict:
    """Compute answer correctness for a Q&A response.

    Args:
        response: The LLM's response
        ground_truth: dict with 'answer' and optionally 'supporting_quote'
        context: The original context paragraph

    Returns:
        dict with:
        - semantic_similarity: float (embedding similarity to ground truth answer)
        - keyword_overlap: float (keyword overlap with ground truth)
        - citation_present: bool
        - citation_accuracy: float
        - overall_score: float (weighted combination)
    """
    if response.startswith("ERROR:"):
        return {
            "semantic_similarity": 0.0,
            "keyword_overlap": 0.0,
            "citation_present": False,
            "citation_accuracy": 0.0,
            "overall_score": 0.0,
        }

    gt_answer = ground_truth.get("answer", "")

    if not gt_answer:
        return {
            "semantic_similarity": 0.0,
            "keyword_overlap": 0.0,
            "citation_present": False,
            "citation_accuracy": 0.0,
            "overall_score": 0.0,
        }

    # Semantic similarity
    embeddings = get_embeddings([response, gt_answer])
    sem_sim = cosine_similarity(embeddings[0], embeddings[1])

    # Keyword overlap
    kw_overlap = keyword_overlap(response, gt_answer)

    # Citation check
    citation = check_citation_present(response, context) if context else {"has_quote": False, "quote_accuracy": 0.0}

    # Weighted overall score
    # 40% semantic similarity + 30% keyword overlap + 30% citation
    citation_score = 0.5 * float(citation["has_quote"]) + 0.5 * citation["quote_accuracy"]
    overall = 0.4 * sem_sim + 0.3 * kw_overlap + 0.3 * citation_score

    return {
        "semantic_similarity": float(sem_sim),
        "keyword_overlap": float(kw_overlap),
        "citation_present": citation["has_quote"],
        "citation_accuracy": citation["quote_accuracy"],
        "overall_score": float(overall),
    }
