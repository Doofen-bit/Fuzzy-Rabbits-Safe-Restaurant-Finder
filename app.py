"""Fuzzy-Rabbits Recipe Finder — Streamlit Web App.

Features:
- Describe ingredients by text or voice upload → semantic recipe retrieval
- Auto-classifies dietary fit (vegan, keto, gluten-free)
- Shows aggregated user review perspectives (positive/negative themes)
- Fusion Mode: finds similar recipes from different cuisines with ingredient swap suggestions
"""

import os
import sys
import tempfile

import streamlit as st

# Ensure src package is importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from src.dietary import get_dietary_badges
from src.fusion import get_fusion_suggestions
from src.reviews import aggregate_reviews
from src.search import load_recipes, search_recipes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fuzzy-Rabbits Recipe Finder",
    page_icon="🐰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .recipe-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #ff6b6b;
    }
    .recipe-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d3436;
        margin-bottom: 0.3rem;
    }
    .cuisine-badge {
        display: inline-block;
        background: #dfe6e9;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        color: #636e72;
        margin-bottom: 0.5rem;
    }
    .dietary-badge {
        display: inline-block;
        background: #55efc4;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        color: #00b894;
        margin-right: 4px;
        font-weight: 600;
    }
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        margin-bottom: 0.5rem;
    }
    .theme-positive {
        color: #00b894;
        font-weight: 600;
    }
    .theme-negative {
        color: #d63031;
        font-weight: 600;
    }
    .fusion-card {
        background: linear-gradient(135deg, #f093fb22, #f5576c22);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #a29bfe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: About
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐰 Fuzzy-Rabbits")
    st.caption("ML Recipe Finder — NYU Project")
    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        1. Describe ingredients (text or voice)
        2. We semantically retrieve matching recipes
        3. See dietary classifications & review themes
        4. Try **Fusion Mode** for cross-cuisine ideas!
        """
    )
    st.markdown("---")
    num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("Built with ❤️ by Fuzzy-Rabbits Team")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🐰 Fuzzy-Rabbits Recipe Finder")
st.subheader("Describe your ingredients → get recipes tailored to you")

# ---------------------------------------------------------------------------
# Input section: Text or Voice
# ---------------------------------------------------------------------------
input_col, voice_col = st.columns([3, 1])

with input_col:
    user_query = st.text_area(
        "📝 Describe your ingredients:",
        placeholder="e.g. 'I have chicken, garlic, lemon, and fresh herbs'",
        height=100,
    )

with voice_col:
    st.markdown("#### 🎙️ Voice Input")
    audio_file = st.file_uploader(
        "Upload audio (WAV/MP3):",
        type=["wav", "mp3", "ogg", "flac"],
        help="Record yourself describing your ingredients and upload the audio file.",
    )
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        with st.spinner("Transcribing audio…"):
            try:
                import speech_recognition as sr

                recognizer = sr.Recognizer()
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                with sr.AudioFile(tmp_path) as source:
                    audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
                st.success(f"Transcribed: *{transcript}*")
                if not user_query:
                    user_query = transcript
            except Exception as exc:
                st.warning(
                    f"Could not transcribe audio automatically ({exc}). "
                    "Please type your ingredients above."
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
col_search, col_diet = st.columns([2, 1])
with col_search:
    search_clicked = st.button("🔍 Find Recipes", type="primary", use_container_width=True)
with col_diet:
    diet_filter = st.multiselect(
        "Filter by diet:",
        ["🌱 Vegan", "🥩 Keto", "🌾 Gluten-Free"],
        default=[],
    )

# ---------------------------------------------------------------------------
# Load recipes and run search
# ---------------------------------------------------------------------------
recipes = load_recipes()

if search_clicked or user_query:
    query = user_query.strip() if user_query else ""
    results = search_recipes(query, recipes=recipes, top_k=num_results)

    # Apply dietary filter
    if diet_filter:
        filter_flags = {
            "🌱 Vegan": "vegan",
            "🥩 Keto": "keto",
            "🌾 Gluten-Free": "gluten_free",
        }
        filtered = []
        for recipe, score in results:
            badges = get_dietary_badges(recipe)
            if all(f in badges for f in diet_filter):
                filtered.append((recipe, score))
        results = filtered

    if not results:
        st.info("No recipes matched your query and dietary filters. Try different ingredients or remove filters.")
    else:
        st.markdown(f"### 🍽️ Found {len(results)} recipe(s)")
        st.markdown("---")

        for recipe, score in results:
            badges = get_dietary_badges(recipe)
            review_data = aggregate_reviews(recipe)

            with st.container():
                # Recipe header
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(
                        f'<div class="recipe-title">{recipe["name"]}</div>'
                        f'<span class="cuisine-badge">🌍 {recipe["cuisine"]}</span>',
                        unsafe_allow_html=True,
                    )
                    if badges:
                        st.markdown(
                            " ".join(
                                f'<span class="dietary-badge">{b}</span>' for b in badges
                            ),
                            unsafe_allow_html=True,
                        )
                with right:
                    if score > 0:
                        pct = min(int(score * 100 / 0.5 * 100), 100)
                        st.metric("Match", f"{min(int(score * 200), 100)}%")

                # Description
                st.markdown(f"*{recipe['description']}*")

                # Expandable details
                with st.expander("📋 Ingredients & Instructions"):
                    ing_col, inst_col = st.columns(2)
                    with ing_col:
                        st.markdown("**Ingredients:**")
                        for ing in recipe["ingredients"]:
                            st.markdown(f"- {ing.title()}")
                    with inst_col:
                        st.markdown("**Instructions:**")
                        st.markdown(recipe.get("instructions", ""))

                # Review perspectives
                with st.expander("💬 Review Perspectives"):
                    rev_col1, rev_col2 = st.columns(2)
                    with rev_col1:
                        pos_count = review_data["positive_count"]
                        st.markdown(
                            f'<span class="theme-positive">👍 {pos_count} positive review(s)</span>',
                            unsafe_allow_html=True,
                        )
                        for theme in review_data["positive_themes"]:
                            st.markdown(f"  ✅ {theme}")
                    with rev_col2:
                        neg_count = review_data["negative_count"]
                        st.markdown(
                            f'<span class="theme-negative">👎 {neg_count} negative review(s)</span>',
                            unsafe_allow_html=True,
                        )
                        for theme in review_data["negative_themes"]:
                            st.markdown(f"  ⚠️ {theme}")

                    # Show all reviews
                    if st.checkbox(f"Show all reviews", key=f"reviews_{recipe['id']}"):
                        for rev in recipe.get("reviews", []):
                            st.markdown(f"> **{rev['author']}**: {rev['text']}")

                # Fusion Mode
                with st.expander("🌐 Fusion Mode — Cross-Cuisine Inspiration"):
                    st.markdown(
                        f"Recipes from **other cuisines** similar to *{recipe['name']}* "
                        f"({recipe['cuisine']}), with ingredient swap ideas:"
                    )
                    fusions = get_fusion_suggestions(recipe, recipes, top_k=3)
                    for fusion in fusions:
                        frec = fusion["recipe"]
                        fbadges = get_dietary_badges(frec)
                        st.markdown(
                            f'<div class="fusion-card">'
                            f'<strong>{frec["name"]}</strong> '
                            f'<span style="color:#a29bfe">({frec["cuisine"]})</span>',
                            unsafe_allow_html=True,
                        )
                        if fbadges:
                            st.markdown(" · ".join(fbadges))
                        st.markdown(f"*{frec['description']}*")
                        swaps = fusion["swaps"]
                        if swaps:
                            st.markdown("**💡 Ingredient swap ideas to blend cuisines:**")
                            for sw in swaps:
                                st.markdown(
                                    f"- Swap **{sw['original']}** → **{sw['swap']}** "
                                    f"*(_{sw['reason']}_)*"
                                )
                        st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem; color: #636e72;">
            <div style="font-size: 4rem;">🥕 🧅 🧄 🍅</div>
            <h3>What's in your fridge?</h3>
            <p>Describe your ingredients above and we'll find recipes that match.<br>
            Try: <em>"chicken, lemon, garlic, and fresh herbs"</em> or 
            <em>"I have spinach, chickpeas and coconut milk"</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
