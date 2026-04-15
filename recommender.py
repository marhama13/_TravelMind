"""
recommender.py
Hybrid Recommendation Engine combining:
  1. Rule-Based Filtering   – budget / mood gates
  2. Content-Based Filtering – TF-IDF cosine similarity on text features
  3. KNN Similarity          – k-nearest neighbours in feature space
  4. User-Based Collaborative Filtering – user-item matrix (when ratings data exists)
  5. Hybrid Score            – weighted combination of all available signals
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from data_loader import MOOD_KEYWORDS, COUNTRY_BUDGET_TIER, BUDGET_MULTIPLIER


# ── 1. Rule-Based Filtering ────────────────────────────────────────────────────

def rule_based_filter(attractions: pd.DataFrame, country: str, mood: str) -> pd.DataFrame:
    """Hard-filter attractions by country and mood keywords."""
    df = attractions[attractions["country"] == country].copy()
    if df.empty:
        return df

    keywords = MOOD_KEYWORDS.get(mood, [])
    if keywords:
        pattern = "|".join(keywords)
        mask = df["category"].str.contains(pattern, case=False, na=False)
        mood_df = df[mask]
        # Fallback: if mood filter too aggressive, return all for that country
        return mood_df if len(mood_df) >= 3 else df
    return df


def budget_filter_cost(cost_row: pd.Series, budget: str) -> dict:
    """Adjust raw costs by budget level multiplier."""
    mult = BUDGET_MULTIPLIER[budget]
    return {
        "stay_cost":         round(cost_row["stay_cost"] * mult, 2),
        "food_cost":         round(cost_row["food_cost"] * mult, 2),
        "local_travel_cost": round(cost_row["local_travel_cost"] * mult, 2),
    }


# ── 2. Content-Based Filtering ────────────────────────────────────────────────

def build_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined text feature column from available columns."""
    df = df.copy()
    df["content_text"] = (
        df["category"].fillna("") + " " +
        df["name"].fillna("") + " " +
        df["city"].fillna("")
    ).str.lower()
    return df


def content_based_scores(df: pd.DataFrame, mood: str, top_n: int = 20) -> pd.DataFrame:
    """
    Rank attractions by cosine similarity of their content text to mood keywords.
    Returns df with a 'cb_score' column, sorted descending.
    """
    if df.empty:
        return df

    df = build_content_features(df)
    mood_query = " ".join(MOOD_KEYWORDS.get(mood, ["travel"]))

    corpus = df["content_text"].tolist() + [mood_query]
    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        tfidf_matrix = tfidf.fit_transform(corpus)
    except ValueError:
        df["cb_score"] = 0.0
        return df

    query_vec = tfidf_matrix[-1]
    item_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(query_vec, item_vecs).flatten()

    df = df.copy()
    df["cb_score"] = sims
    return df.sort_values("cb_score", ascending=False).head(top_n)


# ── 3. KNN Similarity ─────────────────────────────────────────────────────────

def knn_similar_places(df: pd.DataFrame, seed_idx: int = 0, k: int = 10) -> pd.DataFrame:
    """
    Find k most similar attractions using KNN on lat/lon features.
    Returns df with added 'knn_score' column.
    """
    if len(df) < 2:
        df["knn_score"] = 1.0
        return df

    coords = df[["latitude", "longitude"]].values
    scaler = MinMaxScaler()
    coords_scaled = scaler.fit_transform(coords)

    n_neighbors = min(k + 1, len(df))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(coords_scaled)

    seed_point = coords_scaled[min(seed_idx, len(coords_scaled) - 1)].reshape(1, -1)
    distances, indices = knn.kneighbors(seed_point)

    # Convert distance to similarity score (inverse)
    max_dist = distances.flatten().max() + 1e-9
    sim_scores = 1 - (distances.flatten() / max_dist)

    knn_score_map = {idx: score for idx, score in zip(indices.flatten(), sim_scores)}
    df = df.copy()
    df["knn_score"] = [knn_score_map.get(i, 0.0) for i in range(len(df))]
    return df


# ── 4. User-Based Collaborative Filtering ────────────────────────────────────

def collaborative_filtering_scores(
    df: pd.DataFrame,
    user_ratings: pd.DataFrame,
    target_user_id: int = 1,
) -> pd.DataFrame:
    """
    Compute predicted ratings using User-Based CF (Pearson similarity).
    Adds 'cf_score' to df.
    """
    df = df.copy()
    df["cf_score"] = 0.0

    if user_ratings is None or user_ratings.empty:
        return df

    # Build user-item matrix
    pivot = user_ratings.pivot_table(index="user_id", columns="place_name", values="rating")
    pivot = pivot.fillna(0)

    if target_user_id not in pivot.index:
        # Use mean rating across all users as fallback
        mean_ratings = pivot.mean(axis=0)
        name_col = df["name"].str.strip()
        df["cf_score"] = name_col.map(mean_ratings).fillna(0.0)
        # Normalise to 0-1
        max_v = df["cf_score"].max()
        if max_v > 0:
            df["cf_score"] = df["cf_score"] / max_v
        return df

    # Pearson similarity between target user and all others
    target_row = pivot.loc[target_user_id]
    similarities = {}
    for uid in pivot.index:
        if uid == target_user_id:
            continue
        other_row = pivot.loc[uid]
        common = (target_row != 0) & (other_row != 0)
        if common.sum() < 2:
            similarities[uid] = 0.0
            continue
        t = target_row[common].values
        o = other_row[common].values
        denom = (np.std(t) * np.std(o))
        if denom == 0:
            similarities[uid] = 0.0
        else:
            similarities[uid] = np.corrcoef(t, o)[0, 1]

    # Weighted average predicted rating for each place
    predicted = {}
    for place in pivot.columns:
        num, den = 0.0, 0.0
        for uid, sim in similarities.items():
            if sim <= 0:
                continue
            rating = pivot.loc[uid, place]
            if rating > 0:
                num += sim * rating
                den += abs(sim)
        predicted[place] = (num / den) if den > 0 else 0.0

    name_col = df["name"].str.strip()
    df["cf_score"] = name_col.map(predicted).fillna(0.0)
    max_v = df["cf_score"].max()
    if max_v > 0:
        df["cf_score"] = df["cf_score"] / max_v
    return df


# ── 5. Hybrid Score ───────────────────────────────────────────────────────────

def hybrid_recommend(
    attractions: pd.DataFrame,
    user_ratings: pd.DataFrame,
    country: str,
    mood: str,
    budget: str,
    top_n: int = 20,
    cf_user_id: int = 1,
) -> pd.DataFrame:
    """
    Full hybrid pipeline:
      Rule-Based → Content-Based → KNN → Collaborative Filtering → Weighted Hybrid Score
    """
    # Step 1: Rule-Based hard filter
    df = rule_based_filter(attractions, country, mood)
    if df.empty:
        # Fallback: all attractions for country
        df = attractions[attractions["country"] == country].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)

    # Step 2: Content-Based scores
    df = content_based_scores(df, mood, top_n=min(200, len(df)))

    # Step 3: KNN scores
    df = knn_similar_places(df, seed_idx=0, k=min(15, len(df) - 1))

    # Step 4: Collaborative Filtering scores
    df = collaborative_filtering_scores(df, user_ratings, target_user_id=cf_user_id)

    # Step 5: Weighted Hybrid Score
    # Weights: CB=0.40, KNN=0.25, CF=0.35 (CF weight increases if ratings data rich)
    has_cf = user_ratings is not None and not user_ratings.empty
    w_cb  = 0.45 if not has_cf else 0.35
    w_knn = 0.30 if not has_cf else 0.25
    w_cf  = 0.00 if not has_cf else 0.40

    df["hybrid_score"] = (
        w_cb  * df["cb_score"].fillna(0) +
        w_knn * df["knn_score"].fillna(0) +
        w_cf  * df["cf_score"].fillna(0)
    )

    df = df.sort_values("hybrid_score", ascending=False)
    return df.head(top_n).reset_index(drop=True)


# ── Itinerary Builder ─────────────────────────────────────────────────────────

def build_itinerary(
    recommended: pd.DataFrame,
    restaurants: pd.DataFrame,
    local_food: pd.DataFrame,
    country: str,
    num_days: int,
    mood: str,
) -> list[dict]:
    """
    Generate a day-wise itinerary dict list.
    Each day: morning attraction, afternoon attraction, dinner restaurant, local food tip.
    """
    if recommended.empty:
        return []

    attractions_list = recommended["name"].tolist()
    rest_df = restaurants[restaurants["country"] == country].copy()
    rest_list = rest_df["name"].dropna().tolist() if not rest_df.empty else ["Local Restaurant"]
    food_df = local_food[local_food["country"] == country]
    foods = food_df["dish"].tolist() if not food_df.empty else ["Local Cuisine"]

    # Cycle through with variety
    itinerary = []
    attr_idx = 0
    for day in range(1, num_days + 1):
        morning = attractions_list[attr_idx % len(attractions_list)] if attractions_list else "Explore City"
        attr_idx += 1
        afternoon = attractions_list[attr_idx % len(attractions_list)] if len(attractions_list) > 1 else "Leisure Time"
        attr_idx += 1

        rest = rest_list[(day - 1) % len(rest_list)]
        food = foods[(day - 1) % len(foods)]

        itinerary.append({
            "Day": day,
            "Morning": morning,
            "Afternoon": afternoon,
            "Dinner At": rest,
            "Local Food to Try": food,
        })

    return itinerary
