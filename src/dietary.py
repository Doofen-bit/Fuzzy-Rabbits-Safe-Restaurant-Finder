"""Dietary classification for recipes.

Classifies each recipe as vegan, keto, and/or gluten-free based on
ingredient analysis and existing tags.
"""

from typing import Dict, List

# Ingredients that disqualify a recipe from being vegan
_NON_VEGAN = {
    "meat", "beef", "chicken", "pork", "lamb", "turkey", "shrimp", "salmon",
    "fish", "tuna", "anchovy", "anchovies", "eggs", "egg", "milk", "cream",
    "butter", "cheese", "parmesan", "feta", "mozzarella", "yogurt",
    "heavy cream", "sour cream", "whey", "honey", "gelatin", "lard",
    "pancetta", "bacon", "prosciutto", "fish sauce",
}

# Ingredients that disqualify a recipe from being keto (high-carb)
_NON_KETO = {
    "sugar", "flour", "bread", "pasta", "spaghetti", "rice", "basmati rice",
    "noodles", "potato", "sweet potato", "oats", "quinoa", "beans", "lentils",
    "chickpeas", "corn", "tortillas", "pita", "flatbread", "sourdough",
    "arborio rice", "tamarind paste", "mirin", "sake", "brown sugar",
    "red lentils", "peas", "carrots",
}

# Ingredients that contain gluten
_GLUTEN_INGREDIENTS = {
    "flour", "bread", "pasta", "spaghetti", "wheat", "barley", "rye",
    "sourdough", "pita", "flatbread", "soy sauce", "tortillas",
}

# Grain-based ingredients that are naturally gluten-free
_GLUTEN_FREE_GRAINS = {
    "quinoa", "rice", "basmati rice", "rice noodles", "corn tortillas",
    "arborio rice", "cauliflower",
}


def classify_dietary(recipe: Dict) -> Dict[str, bool]:
    """Return a dict of dietary flags for *recipe*.

    Returns
    -------
    dict with keys ``vegan``, ``keto``, ``gluten_free``.
    """
    ingredients_lower = {i.lower() for i in recipe.get("ingredients", [])}
    existing_tags = {t.lower() for t in recipe.get("tags", [])}

    def _matches(ingredient_set: set, keywords: set) -> bool:
        """Return True if any ingredient contains or is contained by a keyword."""
        for ing in ingredient_set:
            for kw in keywords:
                if kw in ing or ing in kw:
                    return True
        return False

    # Vegan: no animal products in ingredients
    is_vegan = "vegan" in existing_tags or not _matches(ingredients_lower, _NON_VEGAN)

    # Keto: low carb – no high-carb ingredients
    is_keto = "keto" in existing_tags or not _matches(ingredients_lower, _NON_KETO)

    # Gluten-free: no gluten-containing ingredients
    has_gluten = _matches(ingredients_lower, _GLUTEN_INGREDIENTS)
    is_gluten_free = "gluten-free" in existing_tags or not has_gluten

    return {"vegan": is_vegan, "keto": is_keto, "gluten_free": is_gluten_free}


def get_dietary_badges(recipe: Dict) -> List[str]:
    """Return list of dietary label strings for display."""
    flags = classify_dietary(recipe)
    badges = []
    if flags["vegan"]:
        badges.append("🌱 Vegan")
    if flags["keto"]:
        badges.append("🥩 Keto")
    if flags["gluten_free"]:
        badges.append("🌾 Gluten-Free")
    return badges
