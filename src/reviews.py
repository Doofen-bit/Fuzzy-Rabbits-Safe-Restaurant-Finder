"""Aggregate user review perspectives for recipes.

Uses TextBlob polarity to classify each review sentence as positive or
negative, then surfaces the most common positive and negative themes.
"""

from typing import Dict, List, Tuple

from textblob import TextBlob

# Keyword → theme mapping for surfacing common themes
_THEME_KEYWORDS: Dict[str, str] = {
    # Positive themes
    "flavorful": "Great flavor",
    "delicious": "Delicious",
    "tasty": "Tasty",
    "amazing": "Amazing",
    "perfect": "Perfect execution",
    "creamy": "Creamy texture",
    "crispy": "Great crispiness",
    "healthy": "Healthy choice",
    "easy": "Easy to make",
    "quick": "Quick to prepare",
    "satisfying": "Satisfying meal",
    "fresh": "Fresh ingredients",
    "authentic": "Authentic taste",
    "fragrant": "Wonderfully fragrant",
    "comforting": "Comfort food",
    "filling": "Very filling",
    # Negative themes
    "salty": "Too salty",
    "bland": "Lacking flavor",
    "dry": "Dry texture",
    "difficult": "Difficult technique",
    "labor": "Labor intensive",
    "expensive": "Expensive ingredients",
    "time": "Time consuming",
    "overrated": "Overrated",
}


def _classify_sentiment(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' for *text*."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return "positive"
    if polarity < -0.05:
        return "negative"
    return "neutral"


def _extract_themes(texts: List[str]) -> List[str]:
    """Return up to 3 theme labels mentioned across *texts*."""
    found: Dict[str, int] = {}
    for text in texts:
        lower = text.lower()
        for keyword, theme in _THEME_KEYWORDS.items():
            if keyword in lower:
                found[theme] = found.get(theme, 0) + 1
    # Sort by frequency, return top 3
    sorted_themes = sorted(found, key=lambda t: found[t], reverse=True)
    return sorted_themes[:3]


def aggregate_reviews(recipe: Dict) -> Dict[str, List[str]]:
    """Aggregate review perspectives for *recipe*.

    Returns
    -------
    dict with keys:
        ``positive_themes``: list of positive theme strings.
        ``negative_themes``: list of negative theme strings.
        ``positive_count``: number of positive reviews.
        ``negative_count``: number of negative reviews.
    """
    reviews = recipe.get("reviews", [])
    positive_texts: List[str] = []
    negative_texts: List[str] = []

    for review in reviews:
        text = review.get("text", "")
        sentiment = _classify_sentiment(text)
        if sentiment == "positive":
            positive_texts.append(text)
        elif sentiment == "negative":
            negative_texts.append(text)

    return {
        "positive_themes": _extract_themes(positive_texts),
        "negative_themes": _extract_themes(negative_texts),
        "positive_count": len(positive_texts),
        "negative_count": len(negative_texts),
    }
