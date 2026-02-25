"""Unit tests for Fuzzy-Rabbits recipe app modules."""

import os
import sys

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dietary import classify_dietary, get_dietary_badges
from src.fusion import get_fusion_suggestions, suggest_ingredient_swaps
from src.reviews import aggregate_reviews, _classify_sentiment
from src.search import load_recipes, search_recipes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recipes():
    return load_recipes()


@pytest.fixture
def vegan_recipe():
    return {
        "id": 99,
        "name": "Test Vegan Bowl",
        "cuisine": "American",
        "description": "A simple vegan bowl.",
        "ingredients": ["quinoa", "chickpeas", "spinach", "olive oil", "lemon"],
        "tags": [],
        "reviews": [],
    }


@pytest.fixture
def keto_recipe():
    return {
        "id": 100,
        "name": "Test Keto Dish",
        "cuisine": "American",
        "description": "A keto dish.",
        "ingredients": ["chicken breast", "butter", "garlic", "heavy cream", "parmesan"],
        "tags": [],
        "reviews": [],
    }


@pytest.fixture
def gluten_recipe():
    return {
        "id": 101,
        "name": "Test Pasta",
        "cuisine": "Italian",
        "description": "Pasta dish.",
        "ingredients": ["pasta", "tomatoes", "garlic", "olive oil"],
        "tags": [],
        "reviews": [],
    }


# ---------------------------------------------------------------------------
# search.py
# ---------------------------------------------------------------------------

class TestSearch:
    def test_load_recipes_returns_list(self, recipes):
        assert isinstance(recipes, list)
        assert len(recipes) > 0

    def test_load_recipes_have_required_fields(self, recipes):
        for recipe in recipes:
            assert "id" in recipe
            assert "name" in recipe
            assert "ingredients" in recipe
            assert "cuisine" in recipe

    def test_search_returns_top_k_results(self, recipes):
        results = search_recipes("chicken garlic lemon", recipes=recipes, top_k=3)
        assert len(results) == 3

    def test_search_result_format(self, recipes):
        results = search_recipes("pasta eggs cheese", recipes=recipes, top_k=5)
        for recipe, score in results:
            assert isinstance(recipe, dict)
            assert isinstance(score, float)
            assert score >= 0.0

    def test_search_ranks_relevant_recipe_higher(self, recipes):
        results = search_recipes("salmon teriyaki soy sauce", recipes=recipes, top_k=5)
        top_recipe = results[0][0]
        assert "salmon" in top_recipe["name"].lower() or "teriyaki" in top_recipe["name"].lower()

    def test_search_with_empty_query(self, recipes):
        results = search_recipes("", recipes=recipes, top_k=5)
        assert len(results) == 5

    def test_search_pasta_query(self, recipes):
        results = search_recipes("pasta carbonara eggs pancetta", recipes=recipes, top_k=3)
        names = [r["name"] for r, _ in results]
        assert "Spaghetti Carbonara" in names

    def test_search_vegan_query(self, recipes):
        results = search_recipes("chickpeas spinach coconut", recipes=recipes, top_k=3)
        names = [r["name"] for r, _ in results]
        # At least one vegan recipe should be in top 3
        assert any("Curry" in n or "Lentil" in n or "Buddha" in n for n in names)


# ---------------------------------------------------------------------------
# dietary.py
# ---------------------------------------------------------------------------

class TestDietary:
    def test_vegan_recipe_classified_vegan(self, vegan_recipe):
        flags = classify_dietary(vegan_recipe)
        assert flags["vegan"] is True

    def test_non_vegan_recipe(self):
        recipe = {
            "ingredients": ["chicken breast", "garlic", "olive oil"],
            "tags": [],
        }
        flags = classify_dietary(recipe)
        assert flags["vegan"] is False

    def test_keto_recipe_no_carbs(self, keto_recipe):
        flags = classify_dietary(keto_recipe)
        assert flags["keto"] is True

    def test_non_keto_recipe_with_pasta(self, gluten_recipe):
        flags = classify_dietary(gluten_recipe)
        assert flags["keto"] is False

    def test_gluten_free_recipe(self, vegan_recipe):
        flags = classify_dietary(vegan_recipe)
        assert flags["gluten_free"] is True

    def test_gluten_recipe_not_gluten_free(self, gluten_recipe):
        flags = classify_dietary(gluten_recipe)
        assert flags["gluten_free"] is False

    def test_existing_tags_respected(self):
        recipe = {
            "ingredients": ["chicken", "butter"],
            "tags": ["vegan", "keto", "gluten-free"],
        }
        flags = classify_dietary(recipe)
        assert flags["vegan"] is True
        assert flags["keto"] is True
        assert flags["gluten_free"] is True

    def test_get_dietary_badges_returns_list(self, vegan_recipe):
        badges = get_dietary_badges(vegan_recipe)
        assert isinstance(badges, list)

    def test_vegan_badge_in_badges(self, vegan_recipe):
        badges = get_dietary_badges(vegan_recipe)
        assert "🌱 Vegan" in badges

    def test_gluten_badge_in_badges(self):
        recipe = {
            "ingredients": ["salmon", "broccoli", "garlic", "olive oil"],
            "tags": [],
        }
        badges = get_dietary_badges(recipe)
        assert "🌾 Gluten-Free" in badges


# ---------------------------------------------------------------------------
# reviews.py
# ---------------------------------------------------------------------------

class TestReviews:
    def test_sentiment_positive(self):
        result = _classify_sentiment("This is absolutely delicious and amazing!")
        assert result == "positive"

    def test_sentiment_negative(self):
        result = _classify_sentiment("This was terrible and horrible.")
        assert result == "negative"

    def test_aggregate_reviews_structure(self, recipes):
        recipe = recipes[0]  # Spaghetti Carbonara
        result = aggregate_reviews(recipe)
        assert "positive_themes" in result
        assert "negative_themes" in result
        assert "positive_count" in result
        assert "negative_count" in result

    def test_aggregate_reviews_counts_correct(self, recipes):
        recipe = recipes[0]
        result = aggregate_reviews(recipe)
        total = result["positive_count"] + result["negative_count"]
        assert total <= len(recipe["reviews"])

    def test_aggregate_reviews_positive_themes_list(self, recipes):
        recipe = recipes[0]
        result = aggregate_reviews(recipe)
        assert isinstance(result["positive_themes"], list)

    def test_aggregate_reviews_negative_themes_list(self, recipes):
        recipe = recipes[0]
        result = aggregate_reviews(recipe)
        assert isinstance(result["negative_themes"], list)

    def test_themes_capped_at_three(self, recipes):
        recipe = recipes[0]
        result = aggregate_reviews(recipe)
        assert len(result["positive_themes"]) <= 3
        assert len(result["negative_themes"]) <= 3

    def test_empty_reviews(self):
        recipe = {"reviews": []}
        result = aggregate_reviews(recipe)
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0


# ---------------------------------------------------------------------------
# fusion.py
# ---------------------------------------------------------------------------

class TestFusion:
    def test_suggest_ingredient_swaps_returns_list(self, recipes):
        recipe = recipes[0]  # Spaghetti Carbonara has butter, garlic etc.
        swaps = suggest_ingredient_swaps(recipe)
        assert isinstance(swaps, list)

    def test_ingredient_swap_has_required_keys(self, recipes):
        # Find a recipe with known swappable ingredients
        recipe = next(r for r in recipes if "soy sauce" in r["ingredients"])
        swaps = suggest_ingredient_swaps(recipe)
        for sw in swaps:
            assert "original" in sw
            assert "swap" in sw
            assert "reason" in sw

    def test_swaps_capped_at_four(self, recipes):
        for recipe in recipes:
            swaps = suggest_ingredient_swaps(recipe)
            assert len(swaps) <= 4

    def test_get_fusion_suggestions_returns_list(self, recipes):
        source = recipes[0]
        results = get_fusion_suggestions(source, recipes, top_k=3)
        assert isinstance(results, list)

    def test_fusion_results_from_different_cuisines(self, recipes):
        source = recipes[0]  # Italian
        results = get_fusion_suggestions(source, recipes, top_k=3)
        source_cuisine = source["cuisine"]
        # At least one result should be from a different cuisine
        cuisines = [r["recipe"]["cuisine"] for r in results]
        assert any(c != source_cuisine for c in cuisines)

    def test_fusion_result_structure(self, recipes):
        source = recipes[0]
        results = get_fusion_suggestions(source, recipes, top_k=2)
        for result in results:
            assert "recipe" in result
            assert "swaps" in result
            assert "similarity_score" in result

    def test_fusion_excludes_source_recipe(self, recipes):
        source = recipes[0]
        results = get_fusion_suggestions(source, recipes, top_k=5)
        result_ids = [r["recipe"]["id"] for r in results]
        assert source["id"] not in result_ids
