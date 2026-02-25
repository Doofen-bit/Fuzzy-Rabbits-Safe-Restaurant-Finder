"""Semantic recipe retrieval using TF-IDF and cosine similarity."""

import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes.json")


def load_recipes() -> List[dict]:
    """Load recipes from the JSON data file."""
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_recipe_documents(recipes: List[dict]) -> List[str]:
    """Combine recipe fields into a single searchable text document per recipe."""
    docs = []
    for r in recipes:
        parts = [
            r["name"],
            r["description"],
            r["cuisine"],
            " ".join(r["ingredients"]),
            r.get("instructions", ""),
        ]
        docs.append(" ".join(parts).lower())
    return docs


def search_recipes(
    query: str,
    recipes: List[dict] | None = None,
    top_k: int = 5,
) -> List[Tuple[dict, float]]:
    """Return the top-k recipes most semantically similar to *query*.

    Parameters
    ----------
    query:
        Free-text ingredient description from the user.
    recipes:
        Optional pre-loaded recipe list; loaded from disk if ``None``.
    top_k:
        Number of results to return.

    Returns
    -------
    List of (recipe, score) tuples, sorted by descending similarity.
    """
    if recipes is None:
        recipes = load_recipes()
    if not query.strip():
        return [(r, 0.0) for r in recipes[:top_k]]

    docs = _build_recipe_documents(recipes)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(docs)

    query_vec = vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(recipes[i], float(scores[i])) for i in top_indices]
