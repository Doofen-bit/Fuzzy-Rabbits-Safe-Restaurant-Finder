"""Fusion Mode: retrieve similar recipes from different cuisines and suggest
ingredient swaps to blend flavors across culinary traditions.
"""

from typing import Dict, List, Tuple

from src.search import _build_recipe_documents, search_recipes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ingredient swap map: ingredient → (swap, reason)
_INGREDIENT_SWAPS: Dict[str, Tuple[str, str]] = {
    "soy sauce": ("fish sauce", "adds umami depth from Southeast Asian cuisine"),
    "fish sauce": ("soy sauce", "vegan-friendly umami from East Asian cuisine"),
    "butter": ("coconut oil", "tropical richness from South Asian cuisine"),
    "olive oil": ("sesame oil", "nutty depth from East Asian cuisine"),
    "sesame oil": ("olive oil", "Mediterranean lightness"),
    "parmesan": ("nutritional yeast", "vegan umami from modern cuisine"),
    "heavy cream": ("coconut milk", "creamy tropical twist from Indian cuisine"),
    "coconut milk": ("heavy cream", "rich European-style creaminess"),
    "basil": ("cilantro", "bright herbaceous note from Southeast Asian cuisine"),
    "cilantro": ("parsley", "milder herbal note from Mediterranean cuisine"),
    "cumin": ("smoked paprika", "smoky warmth from Spanish cuisine"),
    "paprika": ("curry powder", "warm spiced depth from Indian cuisine"),
    "lemon": ("lime", "bright citrus from Mexican and Thai cuisine"),
    "lime": ("lemon", "classic citrus from Mediterranean cuisine"),
    "pasta": ("rice noodles", "light gluten-free base from Southeast Asian cuisine"),
    "rice noodles": ("zucchini noodles", "low-carb option popular in fusion cuisine"),
    "bread": ("lettuce wraps", "refreshing low-carb wrap from Asian cuisine"),
    "sour cream": ("tahini", "creamy nutty dressing from Middle Eastern cuisine"),
    "cheddar cheese": ("feta cheese", "tangy crumble from Mediterranean cuisine"),
    "garlic": ("ginger", "warm spiced aromatics from Asian cuisine"),
    "oregano": ("za'atar", "herbaceous Middle Eastern spice blend"),
    "red wine vinegar": ("rice vinegar", "mild sweet acidity from East Asian cuisine"),
    "white wine": ("sake", "delicate umami from Japanese cuisine"),
    "arborio rice": ("quinoa", "protein-rich grain from Andean cuisine"),
    "pine nuts": ("peanuts", "nutty crunch from East Asian cuisine"),
}


def _find_fusion_recipes(
    source_recipe: Dict,
    all_recipes: List[Dict],
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:
    """Find structurally similar recipes from different cuisines.

    Parameters
    ----------
    source_recipe:
        The recipe to find fusions for.
    all_recipes:
        Full recipe database.
    top_k:
        Number of fusion suggestions to return.

    Returns
    -------
    List of (recipe, similarity_score) tuples from different cuisines.
    """
    source_cuisine = source_recipe.get("cuisine", "").lower()
    candidates = [r for r in all_recipes if r["id"] != source_recipe["id"]]

    docs = _build_recipe_documents([source_recipe] + candidates)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(docs)

    source_vec = tfidf_matrix[0]
    candidate_matrix = tfidf_matrix[1:]
    scores = cosine_similarity(source_vec, candidate_matrix).flatten()

    # Boost recipes from different cuisines by keeping them, penalize same cuisine slightly
    adjusted_scores = []
    for i, candidate in enumerate(candidates):
        score = scores[i]
        if candidate.get("cuisine", "").lower() == source_cuisine:
            score *= 0.5  # demote same-cuisine results
        adjusted_scores.append(score)

    top_indices = np.argsort(adjusted_scores)[::-1][:top_k]
    return [(candidates[i], float(adjusted_scores[i])) for i in top_indices]


def suggest_ingredient_swaps(recipe: Dict) -> List[Dict[str, str]]:
    """Suggest ingredient swaps to give *recipe* a cross-cuisine fusion twist.

    Returns
    -------
    List of dicts with keys ``original``, ``swap``, ``reason``.
    """
    suggestions = []
    for ingredient in recipe.get("ingredients", []):
        key = ingredient.lower()
        if key in _INGREDIENT_SWAPS:
            swap, reason = _INGREDIENT_SWAPS[key]
            suggestions.append(
                {"original": ingredient, "swap": swap, "reason": reason}
            )
    return suggestions[:4]  # Cap at 4 suggestions per recipe


def get_fusion_suggestions(
    source_recipe: Dict,
    all_recipes: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """Return fusion mode suggestions for *source_recipe*.

    Parameters
    ----------
    source_recipe:
        The recipe selected by the user.
    all_recipes:
        Full recipe database.
    top_k:
        Number of similar cross-cuisine recipes to return.

    Returns
    -------
    List of dicts, each containing a ``recipe`` and ``swaps`` key.
    """
    similar = _find_fusion_recipes(source_recipe, all_recipes, top_k=top_k)
    results = []
    for recipe, score in similar:
        swaps = suggest_ingredient_swaps(recipe)
        results.append(
            {
                "recipe": recipe,
                "similarity_score": score,
                "swaps": swaps,
            }
        )
    return results
