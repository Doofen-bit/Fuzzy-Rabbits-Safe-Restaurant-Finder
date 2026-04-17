# Fuzzy-Rabbits — NYC Safe Restaurant Finder

ML Project — NYU  
- Lisa Popova (yp2541@nyu.edu)
- Wendy Liu (jl14704@nyu.edu)
- Yixuan Du (yd2927@nyu.edu)
- George Liu (jl15266@nyu.edu)
- Yuxi Wu (yw8271@nyu.edu)

---

## Overview

An end-to-end machine-learning dashboard built on **295,995 DOHMH restaurant inspection records** from NYC Open Data. The project is structured as five sequential parts, each adding a new ML capability on top of the shared `restaurants.csv` dataset.

| Part | Tab | What it does |
|---|---|---|
| **1** | Data & Exploration | Three-step preprocessing pipeline · filterable data explorer · interactive NYC map |
| **2** | KNN Classifier | From-scratch K-Nearest Neighbors predicts restaurant grade (A/B/C) from inspection history |
| **3** | Decision Tree Classifier | From-scratch weighted-Gini Decision Tree — fixes every class-imbalance problem in Part 2 |
| **4** | Cuisine Predictor | scikit-learn TF-IDF + Logistic Regression predicts cuisine type from restaurant name alone |
| **5** | Safe Restaurant Route | Reinforcement Learning (Value Iteration) plans a walking route to the nearest safe, cuisine-matching restaurant cluster — shown on a real OpenStreetMap |

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd Fuzzy-Rabbits-Safe-Restaurant-Finder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the raw data (see below — one-time)
# 4. Build restaurants.csv
python -m src.preprocessor

# 5. Launch the dashboard
python -m streamlit run streamlit_app.py
```

Your browser opens `http://localhost:8501` automatically.

---

## Data Setup

### Step 1 — Download the raw DOHMH data

The raw dataset is **not in git** (~150 MB). Each team member downloads it once:

1. Go to: https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data
2. Click **Export → CSV**.
3. Rename the file to exactly:
   ```
   DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv
   ```
4. Place it in `data/`:
   ```
   Fuzzy-Rabbits-Safe-Restaurant-Finder/
   └── data/
       └── DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv
   ```

The `data/` folder is in `.gitignore` — CSV files there will never be committed.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

All required packages:

| Package | Used in |
|---|---|
| `pandas`, `numpy` | All parts |
| `streamlit` | Dashboard |
| `plotly`, `pydeck` | Parts 1–3 charts and maps |
| `scikit-learn` | Parts 4 & 5 (TF-IDF, Logistic Regression) |
| `scipy` | Part 5 (Gaussian blur on reward grid) |
| `folium`, `streamlit-folium` | Part 5 (interactive OpenStreetMap) |
| `requests` | Part 5 (OSRM street-routing API) |

### Step 3 — Build restaurants.csv

```bash
python -m src.preprocessor
```

Reads the raw CSV and writes `data/restaurants.csv` — one row per unique restaurant. Takes ~30–60 seconds on the full dataset. Custom paths:

```bash
python -m src.preprocessor path/to/raw.csv path/to/output.csv
```

---

## Running the Dashboard

### Windows

```
python -m streamlit run streamlit_app.py
```

> If `streamlit` is not on PATH, the `python -m streamlit` form above always works.

### WSL (Windows Subsystem for Linux)

```bash
sudo apt-get update && sudo apt-get install -y python3-pip
cd /mnt/c/Users/<your-username>/Fuzzy-Rabbits-Safe-Restaurant-Finder
pip3 install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your Windows browser after the terminal shows the Network URL.

### macOS / Linux

```bash
pip3 install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

---

## Dataset — `restaurants.csv`

One row per unique restaurant (identified by `camis`). Key columns:

| Column | Description |
|---|---|
| `camis` | Unique restaurant ID |
| `dba` | Restaurant name |
| `boro`, `building`, `street`, `zipcode` | Address |
| `cuisine` | Cuisine type (95 categories) |
| `latitude`, `longitude` | Coordinates |
| `latest_grade` | Most recent letter grade: A, B, or C |
| `latest_grade_encoded` | Numeric: A=3, B=2, C=1 |
| `latest_score` | Inspection score (lower = better) |
| `latest_inspection_date` | Date of most recent inspection |
| `inspection_count` | Number of inspections on record |
| `mean_score` | Mean score across all inspections |
| `min_score` / `max_score` | Score range across all inspections |
| `days_since_last_inspection` | Days since last inspection (as of preprocessing run date) |
| `total_violations` | Total violation rows across all inspections |
| `critical_violations` | Count of critical-flag violations |
| `non_critical_violations` | Count of non-critical violations |
| `unique_violation_codes` | Comma-separated list of all violation codes cited |

Load it in Python:

```python
import pandas as pd
restaurants = pd.read_csv(
    "data/restaurants.csv",
    parse_dates=["latest_grade_date", "latest_inspection_date"]
)
```

---

## Part-by-Part Technical Notes

### Part 1 — Data Pipeline

`src/data_loader.py` loads and cleans the raw CSV (column rename, whitespace strip, type casting).  
`src/preprocessor.py` aggregates violation-level rows into one row per restaurant.

### Part 2 — KNN Grade Classifier (`src/knn_classifier.py`)

- Built from scratch with NumPy only (no scikit-learn).
- Features: `mean_score`, `critical_violations`, `total_violations`, `days_since_last_inspection`.
- Distance metric: cosine similarity on z-score-normalised vectors.
- Temporal train/test split: restaurants sorted by `latest_inspection_date` so the model never sees future data.
- Demonstrates severe class-imbalance: Grade A is ~88% of the data, so KNN almost always predicts A.

### Part 3 — Decision Tree Grade Classifier (`src/decision_tree.py`)

- Built from scratch with NumPy only.
- Weighted Gini impurity: each sample weighted by `n / (n_classes × class_count)` so minority grades B and C contribute equally to splits.
- Engineered features add `min_score`, `max_score`, `score_range`, `inspection_count`, `critical_rate`, `violations_per_inspection`.
- Output: feature importances, top decision rules, full tree sketch.

### Part 4 — Cuisine Type Predictor (`src/cuisine_predictor.py`)

- **Vectoriser**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))` on normalised restaurant names.
- **Classifier**: `LogisticRegression(class_weight="balanced")` for calibrated multi-class probabilities.
- **Split options**: stratified random OR geographic hold-out (entire borough as test set).
- **Output**: top-3 predicted cuisines with probabilities; if the name exists in the dataset the true cuisine is shown.
- Typical top-1 accuracy: ~45% across 75 cuisine classes (random baseline: ~1.3%).

### Part 5 — Safe Restaurant Route Finder (`src/rl_route_finder.py`)

**RL formulation:**

| Component | Details |
|---|---|
| Environment | NYC map: 111 × 113 = 12,543 grid cells (~400 m × 420 m each) |
| State | `(row, col)` — agent's current grid cell |
| Actions | 8 compass directions |
| Reward R(s) | Sum of `safety_score × cuisine_match` for restaurants in the cell, Gaussian-smoothed |
| Proximity bias | Gaussian decay centred at start: nearby clusters beat globally-larger far clusters |
| Cuisine filter | Hard threshold on TF-IDF similarity — only matching cuisine restaurants contribute to reward when a description is provided |
| Discount γ | Derived from the user's walking-time budget (5–20 min); caps the effective search radius |
| Algorithm | Value Iteration (pure NumPy Bellman updates, ~40 ms on the full grid) |

**Street routing:** After VI finds the destination cell, the actual pedestrian route is fetched from the free OSRM public API (`router.project-osrm.org`) — real street-level waypoints. Falls back to a rectilinear grid approximation if offline.

**Food keyword expansion:** 50+ regex mappings (e.g. `"ramen"` → `"japanese noodles asian"`) bridge the gap between dish names and the dataset's generic cuisine labels before TF-IDF vectorisation.

**Walk budget presets:**

| Preset | Distance | γ | Max hops |
|---|---|---|---|
| 5 min | ~375 m | 0.50 | 2 |
| 10 min | ~750 m | 0.70 | 4 |
| 15 min | ~1.1 km | 0.80 | 6 |
| 20 min | ~1.5 km | 0.85 | 8 |

---

## Project Structure

```
Fuzzy-Rabbits-Safe-Restaurant-Finder/
├── data/
│   ├── .gitkeep
│   ├── DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv   # download manually
│   └── restaurants.csv                                                  # generated by preprocessor
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # loads and cleans the raw DOHMH CSV
│   ├── preprocessor.py       # aggregates violation rows → one row per restaurant
│   ├── knn_classifier.py     # Part 2: from-scratch KNN (NumPy only)
│   ├── decision_tree.py      # Part 3: from-scratch weighted Decision Tree (NumPy only)
│   ├── cuisine_predictor.py  # Part 4: TF-IDF + Logistic Regression cuisine classifier
│   └── rl_route_finder.py    # Part 5: Value Iteration RL route planner
├── streamlit_app.py          # five-tab interactive dashboard
├── requirements.txt
└── README.md
```

---

## Git Workflow

```bash
# Before starting — sync with main
git pull origin main

# Create your branch (once)
git checkout -b <your-name>/<feature>

# Stage, commit, push
git add src/my_file.py
git commit -m "Short description of what you did"
git push origin <your-name>/<feature>
```

Open a Pull Request on GitHub. Get one teammate to review before merging.  
**Never commit directly to `main`.**

### Branch naming convention

| Person | Branch |
|---|---|
| Yixuan | `yixuan/data-preprocessing` |
| George | `george/knn-model` |
| Lisa | `lisa/knn-distance-metrics` |
| Yuxi | `yuxi/streamlit-ui` |
| Wendy | `wendy/documentation` |

---

## Quick Git Reference

```bash
git pull origin main              # sync with latest
git checkout -b <branch-name>     # create new branch
git add .                         # stage all changes
git commit -m "message"           # commit
git push origin <branch-name>     # push to GitHub
git status                        # see what changed
git log --oneline                 # see commit history
git diff                          # see uncommitted changes
```
