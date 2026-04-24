# Presentation Script — NYC Safe Restaurant Finder
### Speaker: Parts 1 & 6

---

## INTRO (before handing over to other speakers)

> "Hi everyone. I'm going to walk you through how this project is structured and show you the two parts I built — the data pipeline that makes everything work, and the Ultimate Finder that ties it all together at the end."

*[Open the Streamlit app in the browser. The app should already be running. Navigate to the **Part 1: Data & Exploration** tab.]*

---

## PART 1 — Data & Exploration

### 1.1 — The Data Pipeline

> "Everything in this project starts here. We're working with real New York City restaurant inspection data — about 295,000 violation records from the Department of Health. The raw file captures every single inspection event, every violation code, every score. But that's not what we actually want to work with."

*[Point to the three-step pipeline diagram on screen — the three coloured boxes.]*

> "We run the data through three steps. Step one loads the raw CSV and cleans it up — renames 27 columns to readable names, fixes data types, removes garbage dates. Step two is the key transformation: we collapse all those violation records down to one row per restaurant. That's the `build_restaurant_table` function."

*[Toggle on "Load raw CSV to explore Steps 1 & 2". Wait a second for it to load, then point to the four metric boxes that appear.]*

> "You can see here — 295,000 records, but only about 30,000 unique restaurants. Our final dataset has one clean row per restaurant with everything we need: the latest grade, a violation history summary, mean inspection scores, and geographic coordinates."

*[Toggle it back off, then scroll down to the Step 3 section.]*

> "After preprocessing we end up with about 27,000 restaurants. You can see the grade distribution here — the overwhelming majority are grade A, which creates a class imbalance problem that you'll hear about from the next speakers."

### 1.2 — Data Explorer

*[Scroll down to section 1.2.]*

> "The data explorer lets you filter by borough, grade, cuisine, and score range using the sidebar. You can see the actual records, check missing values, dig into any subset you want."

*[In the sidebar, change the grade filter to only show Grade C restaurants. The table will update.]*

> "For example, if I filter to only Grade C — the worst restaurants — you can see exactly which ones they are, where they are, what cuisine they serve. This is the kind of visibility that actually matters if you're trying to avoid a bad meal."

*[Reset the grade filter back to A, B, C.]*

### 1.3 — NYC Map

*[Scroll down to section 1.3 — the map.]*

> "And here's all of them on a map. Green dots are grade A restaurants, yellow are B, red are C. You can change the colour scheme to a heatmap of critical violations instead — that shows you where inspection problems cluster geographically."

*[Switch the "Colour by" dropdown to "Critical violations (heatmap)".]*

> "You can immediately see hotspots — certain neighbourhoods have much higher concentrations of critical violations. That geographic signal is exactly what the RL route finder in Part 5 and the Ultimate Finder in Part 6 use to steer you away from bad areas."

*[Switch back to "Grade (A/B/C)".]*

> "That's Part 1. Now I'll hand over to [next speaker] who will show you how we use this data to build the KNN and Decision Tree classifiers…"

---

*[Other speakers present Parts 2, 3, 4, 5.]*

---

## PART 6 — Ultimate Restaurant Finder

*[Navigate to the **Part 6: Ultimate Finder** tab. The model should already be loaded — if you see the spinner briefly, that's fine.]*

> "Welcome back. Now that you've seen each individual piece — the data pipeline, KNN, Decision Tree, cuisine predictor, and RL route finder — Part 6 combines all of them into one system."

### 6.1 — The Combined Model (auto-loaded)

*[Point to the green success banner and the four metric boxes.]*

> "The first thing you'll notice is that there's nothing to configure here. The model trains automatically when you open this tab. Under the hood it's blending KNN and Decision Tree predictions — KNN contributes 35%, Decision Tree contributes 65%. We tuned those weights to get the best balance: KNN alone predicts almost everything as Grade A because that's the majority class. Decision Tree recovers the B and C predictions better. Together they give a more calibrated safety score between 1 and 3 for every restaurant."

*[Click the expander "How does blending KNN and Decision Tree help?".]*

> "You can see the grade distribution comparison here. KNN alone predicts Grade A for almost everything. Decision Tree is better spread, but tends to over-predict B and C. The combined model sits between them. The histogram at the bottom shows the actual safety score distribution — most restaurants cluster near 2.8 to 3, which makes sense since most NYC restaurants are Grade A."

*[Close the expander, scroll down.]*

### 6.2 — Setting Your Location

> "Next, you tell it where you are. You can click directly on the map to drop a pin anywhere in NYC, or use the neighbourhood quick-jump to get to Times Square, Brooklyn Heights, wherever."

*[In the neighbourhood dropdown, select "Midtown Manhattan". The blue pin moves on the map.]*

> "I'm going to place myself in Midtown."

### 6.3 — Natural Language Query

*[Scroll to section 6.3.]*

> "This is where Part 6 does something none of the other parts do. You can type anything — a dish, a cuisine style, or even a reference restaurant."

*[Type "spicy ramen" into the query box.]*

> "If I type 'spicy ramen', the system uses TF-IDF character n-gram embeddings to match my query against every restaurant's name, cuisine, and borough. It figures out I'm probably looking for Japanese or Korean restaurants and weights the RL reward grid accordingly."

*[Clear the query, then type "something like Shake Shack".]*

> "Or I can type 'something like Shake Shack' — and the system detects that as a restaurant name reference. It finds Shake Shack in the database and routes me toward restaurants with similar names, cuisine, and neighbourhood profile."

*[Clear that and type "wood-fired pizza" for the demo.]*

### 6.4 & 6.5 — Navigation Mode & Parameters

*[Scroll to sections 6.4 and 6.5.]*

> "There are two navigation modes. Area mode uses Reinforcement Learning — the same Value Iteration algorithm from Part 5 — to find the best cluster of matching restaurants and plan a walking route there. Direct mode skips the RL and ranks every nearby restaurant individually, so you can browse through the ranked list with Next and Previous buttons."

> "The walking budget slider controls how far you're willing to walk. I'll leave it at 15 minutes."

### 6.6 — Running It

*[Make sure query is "wood-fired pizza", mode is "Area Finder", walking budget is "15 min (~1.1 km)", "Exclude Grade C restaurants" is checked. Click **Find Ultimate Route**.]*

> "Let me run it now."

*[Wait for the spinner to finish.]*

### 6.7 — Area Mode Results

*[Point to the five metric boxes at the top.]*

> "The system found a cluster of matching restaurants about a kilometre away — Grade A restaurants predicted to have high safety scores. The route is drawn on the map in orange using real street-level waypoints from the OSRM routing API."

*[Scroll down to the map. Point to the route.]*

> "The blue pin is where I am. The purple flag is the destination cluster. The orange line is my walking route — actual streets, not just a straight line. The green markers are the safe restaurants waiting for me at the destination."

*[Scroll down to section 6.9 — the heatmap.]*

> "This heatmap shows the RL value function — bright red means high expected reward, dark blue means low. The system is essentially asking: if I walk in this direction, what's the best outcome I can expect? The orange route traces the gradient from my location toward the peak."

*[Scroll down to section 6.10 — the table.]*

> "And here's the actual list of recommended restaurants at the destination, sorted by predicted safety score."

### Direct Mode Demo (if time allows)

*[Scroll back up to section 6.4, switch to "Direct Route". Click Find Ultimate Route again.]*

> "In Direct mode, instead of a cluster, we get ranked individual restaurants. Showing number one now. I can hit Next to browse through the options — each one re-routes me there directly."

*[Click Next a couple of times.]*

---

## CLOSING

> "So what Part 6 demonstrates is that each individual component — the data pipeline, the classifiers, the cuisine embeddings, the RL agent — becomes more powerful when they work together. The safety score is richer because it uses model predictions instead of just grade letters. The NLP matching is richer because it understands dish descriptions, cuisine styles, and restaurant references. And the navigation is richer because you can choose between exploring an area or going straight to a specific recommendation."

> "That's the project. Thanks."
