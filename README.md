# Fuzzy-Rabbits — Safe Restaurant Recommendation System
ML Project for
- Lisa Popova (yp2541@nyu.edu) - Wendy Liu (jl14704@nyu.edu) - Yixuan Du (yd2927@nyu.edu) - George Liu (jl15266@nyu.edu) - Yuxi Wu (yw8271@nyu.edu)

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
   safe-restaurant-finder/
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

Here's a complete guide for your group to get started with GitHub collaboration and Claude Code.

---

## Setting Up Your GitHub Collaborative Project from Scratch

### Step 1: One Person Creates the Repository (Team Lead)

One person does this:

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** icon → **New repository**
3. Fill in:
   - Repository name: `safe-restaurant-finder` (or similar)
   - Description: "KNN-based NYC restaurant safety recommendation system"
   - Set to **Public** or **Private**
   - ✅ Check "Add a README file"
   - ✅ Add `.gitignore` → choose **Python**
4. Click **Create repository**

---

### Step 2: Add All Team Members as Collaborators

The repo creator goes to:
**Settings → Collaborators → Add people** → type each teammate's GitHub username

Each teammate will get an email invite — everyone must **accept** before they can push code.

---

### Step 3: Everyone Clones the Repository

Each team member runs this on their own computer:

```bash
git clone https://github.com/<team-lead-username>/safe-restaurant-finder.git
cd safe-restaurant-finder
```

---

### Step 4: Set Up the Project Structure (Team Lead Does This Once)

The team lead creates the initial folder structure and pushes it:

```
safe-restaurant-finder/
│
├── data/                  # Raw and processed data files
│   └── .gitkeep
├── notebooks/             # Jupyter notebooks for exploration
│   └── .gitkeep
├── src/                   # Core Python source code
│   ├── __init__.py
│   ├── data_loader.py     # Yixuan's area
│   ├── preprocessor.py    # Yixuan's area
│   ├── knn_model.py       # George & Lisa's area
│   └── recommender.py     # George & Lisa's area
├── app.py                 # Streamlit app — Yuxi's area
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

Create a `requirements.txt` right away:

```
streamlit
pandas
numpy
requests
plotly
```

Push this structure:

```bash
git add .
git commit -m "Initial project structure"
git push origin main
```

---

### Step 5: Everyone Works on Their Own Branch — The Golden Rule

**Never commit directly to `main`.** Each person works on a feature branch:

```bash
# Pull the latest main first
git pull origin main

# Create and switch to your own branch
git checkout -b yixuan/data-preprocessing
```

Branch naming convention for your team:

| Person | Branch Name |
|---|---|
| Yixuan | `yixuan/data-preprocessing` |
| George | `george/knn-model` |
| Lisa | `lisa/knn-distance-metrics` |
| Yuxi | `yuxi/streamlit-ui` |
| Wendy | `wendy/documentation` |

---

### Step 6: Daily Workflow — What Each Person Does Every Day

```bash
# 1. Before starting work, always sync with main
git pull origin main

# 2. Do your work, then stage your changes
git add src/data_loader.py      # add specific files
# or
git add .                       # add everything

# 3. Commit with a clear message
git commit -m "Add data loading function for NYC Open Data API"

# 4. Push your branch to GitHub
git push origin yixuan/data-preprocessing
```

---

### Step 7: Merging Work — Pull Requests

When a piece of work is ready to be shared with the team:

1. Go to the repo on GitHub
2. You'll see a banner: **"Compare & pull request"** — click it
3. Write a short description of what you did
4. Assign a teammate to **review** it
5. Once approved, click **Merge pull request**
6. Everyone then runs `git pull origin main` to get the update

**Tip:** Do small, frequent pull requests rather than one giant one at the end. It prevents painful merge conflicts.

---

### Step 8: Handling Merge Conflicts

If two people edited the same file, Git will flag a conflict. The file will look like this:

```
<<<<<<< HEAD
your version of the code
=======
teammate's version of the code
>>>>>>> george/knn-model
```

You manually edit the file to keep the right version, then:

```bash
git add <conflicted-file>
git commit -m "Resolve merge conflict in knn_model.py"
```

**Best way to avoid conflicts:** communicate in your group chat before editing shared files like `app.py`.

---

### Running the Streamlit App Locally

Anyone can run the app at any time with:

```bash
streamlit run app.py
```

It opens automatically in your browser at `http://localhost:8501`.

---

## How to Use Claude Code for Help

Claude Code is a command-line tool that gives you an AI coding assistant directly in your terminal, with full access to your project files.

### Installing Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

You need Node.js 18+ installed. Then authenticate:

```bash
claude
```

It will prompt you to log in with your Anthropic account the first time.

### Starting a Session in Your Project

```bash
cd safe-restaurant-finder
claude
```

Claude Code can now see all your files. You talk to it in plain English.

### What You Can Ask Claude Code to Do

Since you're responsible for the **data section**, here are example prompts tailored to your work:

```
Write a function in src/data_loader.py that fetches restaurant 
inspection data from the NYC Open Data API for a given zip code.
```

```
Look at src/preprocessor.py and add a function that groups records 
by CAMIS, computes total violations, critical violation count, and 
days since last inspection.
```

```
The data has missing GRADE values. Update the preprocessor to encode 
grades as A=3, B=2, C=1, ungraded=0.
```

```
Add min-max normalization to the feature vectors in preprocessor.py.
```

```
Run the data loader and show me what the first 5 rows look like after 
preprocessing.
```

### Key Things to Know About Claude Code

- It reads and edits your **actual files** — always be on a branch (not `main`) when working with it so you can review changes before merging
- You can say **"show me what you changed"** and it will diff the edits
- If it makes a mistake, say **"undo that"** or just use `git diff` and `git checkout` to revert
- You can paste in error messages directly and ask it to fix them
- It works best with specific, concrete requests rather than vague ones

### Suggested Workflow with Claude Code

```bash
# 1. Make sure you're on your branch
git checkout yixuan/data-preprocessing

# 2. Open Claude Code
claude

# 3. Ask it to help you write or fix code

# 4. Review the changes it made
git diff

# 5. If happy, commit
git commit -m "Add preprocessing pipeline with feature engineering"
```

---

## Quick Reference Cheat Sheet

```bash
git pull origin main              # Sync with latest
git checkout -b <branch-name>     # Create new branch
git add .                         # Stage all changes
git commit -m "message"           # Commit
git push origin <branch-name>     # Push to GitHub
git status                        # See what's changed
git log --oneline                 # See commit history
git diff                          # See uncommitted changes
```

The most important habits for a 5-person team: **always branch, pull before you start, commit often, and communicate before touching shared files.** Good luck with the project!