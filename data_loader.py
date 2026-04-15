"""
data_loader.py
Loads and preprocesses all datasets for the Travel Recommendation System.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Budget mapping ──────────────────────────────────────────────────────────────
BUDGET_MULTIPLIER = {"Low": 0.6, "Medium": 1.0, "High": 1.6}

# Cost thresholds (monthly INR equivalent for normalisation)
COUNTRY_BUDGET_TIER = {
    "Italy": "Medium", "France": "High", "Spain": "Medium",
    "Germany": "Medium", "India": "Low", "Japan": "High",
    "Thailand": "Low", "United States": "High", "Canada": "High",
    "Australia": "High", "United Arab Emirates": "High", "Turkey": "Low",
    "South Africa": "Low", "Egypt": "Low", "Morocco": "Low",
    "Brazil": "Low", "Mexico": "Low", "Indonesia": "Low",
    "Singapore": "High", "United Kingdom": "High",
    "Netherlands": "Medium", "South Korea": "Medium", "Switzerland": "High",
}

# Mood → category keyword mapping for rule-based filtering
MOOD_KEYWORDS = {
    "Relax":     ["park", "garden", "beach", "lake", "nature", "spa", "viewpoint"],
    "Adventure": ["hike", "volcano", "sport", "climb", "trek", "adventure", "bunker", "cave"],
    "Culture":   ["museum", "temple", "church", "palace", "monument", "historic", "gallery", "cathedral", "basilica", "fort"],
    "Romantic":  ["fountain", "rooftop", "sunset", "cruise", "garden", "tower", "viewpoint", "wine"],
}


def load_attractions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "attractions.csv"))
    df.columns = df.columns.str.strip()
    # Drop rows with no name or coordinates
    df = df[df["name"].notna() & df["latitude"].notna() & df["longitude"].notna()].copy()
    df["name"] = df["name"].str.strip()
    df = df[df["name"] != ""]
    df["category"] = df["category"].fillna("Attraction").str.strip()
    df["country"] = df["country"].str.strip()
    df["city"] = df["city"].str.strip()
    df.reset_index(drop=True, inplace=True)
    return df


def load_restaurants() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"))
    df.columns = df.columns.str.strip()
    df = df[df["name"].notna() & df["latitude"].notna() & df["longitude"].notna()].copy()
    df["name"] = df["name"].str.strip()
    df = df[df["name"] != ""]
    df["cuisine"] = df["cuisine"].fillna("local").str.strip()
    df["country"] = df["country"].str.strip()
    df["city"] = df["city"].str.strip()
    df.reset_index(drop=True, inplace=True)
    return df


def load_local_food() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "local_food.csv"))
    df.columns = df.columns.str.strip()
    df["dish"] = df["dish"].str.strip()
    df["country"] = df["country"].str.strip()
    df["city"] = df["city"].str.strip()
    return df


def load_cost_estimation() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "cost_estimation.csv"))
    df.columns = df.columns.str.strip()
    df["country"] = df["country"].str.strip()
    df["city"] = df["city"].str.strip()
    return df


def load_user_ratings() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "user_ratings.csv"))
    df.columns = df.columns.str.strip()
    df["place_name"] = df["place_name"].str.strip()
    return df


def get_country_city_map(attractions: pd.DataFrame) -> dict:
    """Returns {country: [city, …]} mapping from attractions data."""
    mapping = {}
    for country, grp in attractions.groupby("country"):
        mapping[country] = sorted(grp["city"].unique().tolist())
    return mapping
