# Fuzzy-Rabbits — Safe Restaurant Recommendation System

ML Project for  
- Lisa Popova (yp2541@nyu.edu)
- Wendy Liu (jl14704@nyu.edu)
- Yixuan Du (yd2927@nyu.edu)
- George Liu (jl15266@nyu.edu)
- Yuxi Wu (yw8271@nyu.edu)

---

## Data Setup and Preprocessing (read this first)

### Step 1 — Download the raw data

The raw dataset is **not stored in git** (it is ~150 MB). Each team member must download it manually once:

1. Go to: https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data
2. Click **Export → CSV** to download the file.
3. Rename the downloaded file to match the exact filename expected by the pipeline:
   ```
   DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv
   ```
4. Place it inside the `data/` folder of this repo:
   ```
   Fuzzy-Rabbits-Safe-Restaurant-Finder/
   └── data/
       └── DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv
   ```

The `data/` folder is listed in `.gitignore` — CSV files there will never be accidentally committed.

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Run the preprocessing pipeline

From the **project root**, run:

```bash
python -m src.preprocessor
```

This reads the raw CSV and writes `data/restaurants.csv` — a clean table with **one row per unique restaurant**. It takes about 30–60 seconds on the full dataset.

You can also specify custom input/output paths:

```bash
python -m src.preprocessor path/to/raw.csv path/to/output.csv
```

---

### What `data/restaurants.csv` contains

Each row is one unique restaurant (identified by `camis`). Columns:

| Column | Description |
|---|---|
| `camis` | Unique restaurant ID (use this as the join key) |
| `dba` | Restaurant name |
| `boro`, `building`, `street`, `zipcode` | Address |
| `cuisine` | Cuisine type |
| `latitude`, `longitude` | Coordinates for map/distance features |
| `nta`, `community_board`, `council_district` | Geographic subdivisions |
| `latest_grade` | Most recent letter grade: A, B, or C |
| `latest_grade_encoded` | Numeric: A=3, B=2, C=1 (use as KNN target label) |
| `latest_grade_date` | Date the current grade was issued |
| `latest_score` | Score from the most recent inspection (lower = better) |
| `latest_inspection_date` | Date of the most recent inspection |
| `latest_action` | Action taken at the most recent inspection |
| `inspection_count` | Number of unique inspections on record |
| `mean_score` | Mean score across all inspections |
| `min_score` / `max_score` | Score range across all inspections |
| `days_since_last_inspection` | Days between last inspection and script run date |
| `total_violations` | Total violation records across all inspections |
| `critical_violations` | Count of critical-flag violations |
| `non_critical_violations` | Count of non-critical violations |
| `unique_violation_codes` | Comma-separated list of all violation codes ever cited |

---

### How to load `restaurants.csv` in your code

```python
import pandas as pd

restaurants = pd.read_csv("data/restaurants.csv", parse_dates=["latest_grade_date", "latest_inspection_date"])
```

**For the KNN classifier** (George & Lisa) — the feature matrix and target label are ready to use:

```python
features = restaurants[["mean_score", "latest_score", "critical_violations",
                         "total_violations", "days_since_last_inspection",
                         "inspection_count"]].dropna()
labels = restaurants.loc[features.index, "latest_grade_encoded"]
```

**For the text retrieval / SentenceTransformer** (Wendy) — build a description string per restaurant:

```python
restaurants["text"] = (
    restaurants["cuisine"].fillna("") + " restaurant in " +
    restaurants["boro"].fillna("") + ", " +
    restaurants["street"].fillna("")
)
```

**For the Streamlit map** (Yuxi) — latitude and longitude are numeric and ready:

```python
map_data = restaurants[["dba", "latitude", "longitude", "latest_grade"]].dropna()
```

---

## Running the Streamlit Dashboard

`streamlit_app.py` visualises the full conversion pipeline and lets you explore all 30,935 restaurants on an interactive NYC map. You must complete **Data Setup Steps 1–3** above before running it.

### Windows (native Python)

The `streamlit` command may not be on your PATH after installation. Use the module flag instead — this always works:

```
python -m streamlit run streamlit_app.py
```

Your browser will open `http://localhost:8501` automatically.

> **If you see `'streamlit' is not recognized`**, that means the Scripts folder is not on PATH. The `python -m streamlit` form above bypasses this entirely, so use that.

---

### WSL (Windows Subsystem for Linux)

WSL has a separate Python environment from Windows. Install the dependencies inside WSL first:

```bash
# 1. Install pip if it is not already present
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv

# 2. Navigate to the project (adjust the path if your username differs)
cd /mnt/c/Users/<your-username>/Fuzzy-Rabbits-Safe-Restaurant-Finder

# 3. Install Python dependencies
pip3 install -r requirements.txt

# 4. Run the app
python3 -m streamlit run streamlit_app.py
```

WSL cannot open a browser automatically. After the terminal shows `Network URL: http://localhost:8501`, open that address in your **Windows** browser.

> **Do not run** `python3 pip install …` — that tries to run pip as a script file.  
> The correct form is `pip3 install …` or `python3 -m pip install …`.

---

### macOS / Linux

```bash
pip3 install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

---

### What the dashboard shows

| Tab | Contents |
|---|---|
| **Conversion Pipeline** | Step-by-step view of how raw DOHMH violation records become `restaurants.csv`, with metrics, transformation tables, and charts at each step. Toggle "Load raw data" to inspect the source CSV live. |
| **Data Explorer** | Filterable table of all restaurants. Use the sidebar to narrow by borough, grade, cuisine, and score range. |
| **NYC Map** | Interactive scatter map of ~30,500 restaurants coloured by grade (green = A, yellow = B, red = C). Switch to a heatmap view weighted by critical violations. Click any dot for name, address, grade, and score. |

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
│   ├── data_loader.py       # loads and cleans raw CSV
│   └── preprocessor.py      # aggregates to restaurant level
├── streamlit_app.py         # interactive dashboard
├── requirements.txt
└── README.md
```

---

## Daily Git Workflow

```bash
# Before starting — sync with main
git pull origin main

# Create your branch (do this once)
git checkout -b <your-name>/<feature>

# Stage and commit your work
git add src/my_file.py
git commit -m "Short description of what you did"

# Push to GitHub
git push origin <your-name>/<feature>
```

Then open a Pull Request on GitHub. Get one teammate to review it before merging.

**Branch naming convention:**

| Person | Branch |
|---|---|
| Yixuan | `yixuan/data-preprocessing` |
| George | `george/knn-model` |
| Lisa | `lisa/knn-distance-metrics` |
| Yuxi | `yuxi/streamlit-ui` |
| Wendy | `wendy/documentation` |

**Never commit directly to `main`.**

---

## Quick Reference

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
