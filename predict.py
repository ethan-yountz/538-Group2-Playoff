"""
predict.py
=======================
Fills in the Predictions.csv file with model predictions for all three variables:
  - Total   : Home Points + Away Points  (THIS FILE — Ridge regression, 13 features)
  - Spread  : Home Points - Away Points  (ADD YOUR MODEL IN SECTION B)
  - OREB    : Offensive Rebounds         (ADD YOUR MODEL IN SECTION C)

Usage:
    python predict.py

Output:
    Predictions_filled.csv  — copy of Predictions.csv with the Total column filled in.
                              Once Spread and OREB models are added, all three will be filled.

Requirements:
    pip install pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

MASTER_CSV     = Path(__file__).resolve().parent / "Data" / "final" / "master_games_model_ready_regular_season.csv"
PREDICTIONS_CSV = "Predictions.csv"
OUTPUT_CSV      = "Predictions_filled.csv"

# Load the full historical dataset
df = pd.read_csv(MASTER_CSV, low_memory=False)
df["gameDate"] = pd.to_datetime(df["gameDate"])
df = df[df["gameDate"] >= "2020-07-30"].copy()  # post-bubble filter
df = df[df["Total"] > 0].copy()                 # drop 2 data errors
df = df.sort_values("gameDate").reset_index(drop=True)

# Load the predictions template
pred_df = pd.read_csv(PREDICTIONS_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# TEAM NAME MAPPING
# The training data uses short names (e.g. "Celtics"), the predictions CSV uses
# full names (e.g. "Boston Celtics"). This mapping converts full → short.
# ─────────────────────────────────────────────────────────────────────────────

TEAM_NAME_MAP = {
    "Atlanta Hawks":           "Hawks",
    "Boston Celtics":          "Celtics",
    "Brooklyn Nets":           "Nets",
    "Charlotte Hornets":       "Hornets",
    "Chicago Bulls":           "Bulls",
    "Cleveland Cavaliers":     "Cavaliers",
    "Dallas Mavericks":        "Mavericks",
    "Denver Nuggets":          "Nuggets",
    "Detroit Pistons":         "Pistons",
    "Golden State Warriors":   "Warriors",
    "Houston Rockets":         "Rockets",
    "Indiana Pacers":          "Pacers",
    "Los Angeles Clippers":    "Clippers",
    "Los Angeles Lakers":      "Lakers",
    "Memphis Grizzlies":       "Grizzlies",
    "Miami Heat":              "Heat",
    "Milwaukee Bucks":         "Bucks",
    "Minnesota Timberwolves":  "Timberwolves",
    "New Orleans Pelicans":    "Pelicans",
    "New York Knicks":         "Knicks",
    "Oklahoma City Thunder":   "Thunder",
    "Orlando Magic":           "Magic",
    "Philadelphia 76ers":      "76ers",
    "Phoenix Suns":            "Suns",
    "Portland Trail Blazers":  "Trail Blazers",
    "Sacramento Kings":        "Kings",
    "San Antonio Spurs":       "Spurs",
    "Toronto Raptors":         "Raptors",
    "Utah Jazz":               "Jazz",
    "Washington Wizards":      "Wizards",
}

pred_df["HomeShort"] = pred_df["Home"].map(TEAM_NAME_MAP)
pred_df["AwayShort"] = pred_df["Away"].map(TEAM_NAME_MAP)
pred_df["PredDate"]  = pd.to_datetime(pred_df["Date"].str.replace(r"^\w+,\s*", "", regex=True))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — TOTAL PREDICTION MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Model: Ridge Regression (alpha=100) on 13 Lasso-selected features.
# Train/test split: chronological, trained on games before 2026-01-14.
# Test MAE on held-out data (Jan 14 – Feb 19 2026): 14.70 points.
#
# All features are strictly pre-game (lagged). No current-game information
# is used. Features are standardized using training-set statistics.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION A: Building Total prediction model...")
print("=" * 60)

# ── A1. Build per-team game log ──────────────────────────────────────────────
# Stack every game twice (once per team) so we can compute per-team
# rolling/EWM stats across both home and away appearances.

team_rows = []
for _, row in df.iterrows():
    team_rows.append({
        "gameDate": row["gameDate"],
        "team":     row["HomeTeam"],
        "teamId":   row["HomeTeamId"],
        "pts":      row["HomePTS"],
        "pts_against": row["AwayPTS"],
        "total":    row["Total"],
        "poss":     row["Home_poss"],
    })
    team_rows.append({
        "gameDate": row["gameDate"],
        "team":     row["AwayTeam"],
        "teamId":   row["AwayTeamId"],
        "pts":      row["AwayPTS"],
        "pts_against": row["HomePTS"],
        "total":    row["Total"],
        "poss":     row["Away_poss"],
    })

tdf = (pd.DataFrame(team_rows)
         .sort_values(["team", "gameDate"])
         .reset_index(drop=True))

# ── A2. Per-team rolling/EWM stats (lagged 1 game so no current-game leakage)

def lagged_roll(series, window):
    return series.shift(1).rolling(window, min_periods=3).mean()

def lagged_ewm(series, span):
    return series.shift(1).ewm(span=span, min_periods=3).mean()

tdf["roll20_total"]  = tdf.groupby("team")["total"].transform(lambda x: lagged_roll(x, 20))
tdf["ewm10_total"]   = tdf.groupby("team")["total"].transform(lambda x: lagged_ewm(x, 10))
tdf["ewm20_total"]   = tdf.groupby("team")["total"].transform(lambda x: lagged_ewm(x, 20))
tdf["ewm10_poss"]    = tdf.groupby("team")["poss"].transform(lambda x: lagged_ewm(x, 10))
tdf["ewm20_poss"]    = tdf.groupby("team")["poss"].transform(lambda x: lagged_ewm(x, 20))
tdf["roll10_against"]= tdf.groupby("team")["pts_against"].transform(lambda x: lagged_roll(x, 10))

PERTEAM_COLS = ["roll20_total","ewm10_total","ewm20_total",
                "ewm10_poss","ewm20_poss","roll10_against"]

# ── A3. Merge per-team stats back onto the main game-level dataset ────────────

home_stats = (tdf.rename(columns={"team":"HomeTeam"})
                 .rename(columns={c: f"home_{c}" for c in PERTEAM_COLS})
              [["gameDate","HomeTeam"] + [f"home_{c}" for c in PERTEAM_COLS]])

away_stats = (tdf.rename(columns={"team":"AwayTeam"})
                 .rename(columns={c: f"away_{c}" for c in PERTEAM_COLS})
              [["gameDate","AwayTeam"] + [f"away_{c}" for c in PERTEAM_COLS]])

df = df.merge(home_stats, on=["gameDate","HomeTeam"], how="left")
df = df.merge(away_stats, on=["gameDate","AwayTeam"], how="left")

# ── A4. Combined (home + away) engineered features ────────────────────────────

df["combined_roll20_total"]    = df["home_roll20_total"]   + df["away_roll20_total"]
df["combined_ewm10_total"]     = df["home_ewm10_total"]    + df["away_ewm10_total"]
df["combined_ewm20_total"]     = df["home_ewm20_total"]    + df["away_ewm20_total"]
df["combined_ewm10_poss"]      = df["home_ewm10_poss"]     + df["away_ewm10_poss"]
df["combined_ewm20_poss"]      = df["home_ewm20_poss"]     + df["away_ewm20_poss"]
df["combined_roll10_pts_against"] = df["home_roll10_against"] + df["away_roll10_against"]

# Dataset-derived combined features (these columns already exist in the CSV)
df["avg_last10_pace"] = (df["Home_last10_pts_for"] + df["Home_last10_pts_against"] +
                         df["Away_last10_pts_for"]  + df["Away_last10_pts_against"])
df["opp_adj_total"]   = (df["Home_last10_pts_for"] + df["Away_last10_pts_against"] +
                         df["Away_last10_pts_for"]  + df["Home_last10_pts_against"])
df["home_winpct"]     = df["Home_seasonWins"] / (df["Home_seasonWins"] + df["Home_seasonLosses"] + 1e-9)
df["both_b2b"]        = ((df["Home_b2b"] == 1) & (df["Away_b2b"] == 1)).astype(int)
df["either_b2b"]      = ((df["Home_b2b"] == 1) | (df["Away_b2b"] == 1)).astype(int)

# Head-to-head rolling average Total for this specific team pair
df["pair"] = df.apply(lambda r: tuple(sorted([r["HomeTeamId"], r["AwayTeamId"]])), axis=1)
df["h2h_avg_total"] = df.groupby("pair")["Total"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df["h2h_avg_total"] = df["h2h_avg_total"].fillna(df["Total"].mean())

# ── A5. Define the 13 Lasso-selected features ────────────────────────────────

TOTAL_FEATURES = [
    "combined_roll20_total",       # strongest honest predictor (r=0.315)
    "combined_ewm20_total",        # EWM version, span=20
    "combined_ewm20_poss",         # lagged pace estimate, span=20
    "avg_last10_pace",             # combined scoring pace last 10 games
    "Away_last5_pts_for",          # away team recent offense
    "combined_ewm10_poss",         # lagged pace estimate, span=10
    "h2h_avg_total",               # head-to-head historical Total
    "Home_last10_pts_against",     # home team recent defense
    "Home_b2b",                    # home team on back-to-back (fatigue)
    "home_winpct",                 # home team season win %
    "both_b2b",                    # both teams on back-to-back
    "Home_last5_pts_against",      # home team recent defense (short window)
    "opp_adj_total",               # opponent-adjusted combined scoring
]

# ── A6. Train the model on all historical data (up to prediction date) ────────
# We train on ALL available data (not just pre-Jan-14) since we are now
# generating real future predictions, not evaluating on a held-out set.

train = df.dropna(subset=TOTAL_FEATURES + ["Total"]).copy()

X_train = train[TOTAL_FEATURES].values.astype(float)
y_train = train["Total"].values

# Standardize features — save scaler to apply same transformation to future games
scaler_total = StandardScaler()
X_train_sc   = scaler_total.fit_transform(X_train)

# Ridge regression, alpha=100 (selected by time-series CV in methodology)
model_total = Ridge(alpha=100)
model_total.fit(X_train_sc, y_train)

print(f"  Training games: {len(train)}")
print(f"  Features: {TOTAL_FEATURES}")
print(f"  Intercept: {model_total.intercept_:.2f}")
print(f"  Coefficients: {dict(zip(TOTAL_FEATURES, model_total.coef_.round(3)))}")

# ── A7. Compute features for each future game and predict ─────────────────────

TRAIN_MEAN_TOTAL = train["Total"].mean()  # fallback for missing H2H history

def get_total_features_for_game(game_date, home_team, away_team, history_df, team_log_df):
    """
    Compute all 13 Total features for a single future game.

    Parameters
    ----------
    game_date   : pd.Timestamp  — date of the game to predict
    home_team   : str           — short team name (e.g. "Celtics")
    away_team   : str           — short team name (e.g. "Nets")
    history_df  : pd.DataFrame  — full game history (grows as we predict game by game)
    team_log_df : pd.DataFrame  — per-team game log (grows as we predict game by game)

    Returns
    -------
    np.array of shape (1, 13) — feature vector for this game
    """

    # --- Per-team EWM/rolling stats from team log ---
    def team_stat(team, col, func, **kwargs):
        past = team_log_df[(team_log_df["team"] == team) &
                           (team_log_df["gameDate"] < game_date)].sort_values("gameDate")
        if len(past) < 3:
            return np.nan
        s = past[col]
        if func == "roll":
            return s.rolling(kwargs["w"], min_periods=3).mean().iloc[-1]
        elif func == "ewm":
            return s.ewm(span=kwargs["span"], min_periods=3).mean().iloc[-1]

    home_roll20_total = team_stat(home_team, "total", "roll", w=20)
    away_roll20_total = team_stat(away_team, "total", "roll", w=20)
    home_ewm20_total  = team_stat(home_team, "total", "ewm", span=20)
    away_ewm20_total  = team_stat(away_team, "total", "ewm", span=20)
    home_ewm10_poss   = team_stat(home_team, "poss",  "ewm", span=10)
    away_ewm10_poss   = team_stat(away_team, "poss",  "ewm", span=10)
    home_ewm20_poss   = team_stat(home_team, "poss",  "ewm", span=20)
    away_ewm20_poss   = team_stat(away_team, "poss",  "ewm", span=20)

    combined_roll20_total = (home_roll20_total or 0) + (away_roll20_total or 0)
    combined_ewm20_total  = (home_ewm20_total  or 0) + (away_ewm20_total  or 0)
    combined_ewm10_poss   = (home_ewm10_poss   or 0) + (away_ewm10_poss   or 0)
    combined_ewm20_poss   = (home_ewm20_poss   or 0) + (away_ewm20_poss   or 0)

    # --- Dataset-style rolling stats (last 5 / last 10 games per team) ---
    def team_rolling(team, col, window, is_home):
        """Get last N games average of a column for a team, from game history."""
        if is_home:
            past_home = history_df[(history_df["HomeTeam"] == team) &
                                   (history_df["gameDate"] < game_date)].sort_values("gameDate").tail(window)
            past_away = history_df[(history_df["AwayTeam"] == team) &
                                   (history_df["gameDate"] < game_date)].sort_values("gameDate").tail(window)
            # Map column names for away appearance
            col_map = {"HomePTS": "AwayPTS", "AwayPTS": "HomePTS",
                       "Home_last5_pts_for": None, "Home_last10_pts_against": None}
        vals_home = past_home[col].tolist() if col in past_home.columns else []
        vals_away = past_away[col].tolist() if col in past_away.columns else []
        all_vals = vals_home + vals_away
        if len(all_vals) == 0:
            return np.nan
        return np.mean(all_vals[-window:])

    # Use pre-existing dataset rolling columns from the most recent game for each team
    def last_val(team, col, as_home=True):
        if as_home:
            rows = history_df[(history_df["HomeTeam"] == team) &
                              (history_df["gameDate"] < game_date)].sort_values("gameDate")
        else:
            rows = history_df[(history_df["AwayTeam"] == team) &
                              (history_df["gameDate"] < game_date)].sort_values("gameDate")
        if len(rows) == 0:
            return np.nan
        return rows[col].iloc[-1]

    # For dataset rolling columns, grab the most recent game where the team appeared
    # (these rolling values in the dataset already reflect the team's last N games)
    def best_last_val(team, home_col, away_col):
        home_rows = history_df[(history_df["HomeTeam"] == team) &
                               (history_df["gameDate"] < game_date)].sort_values("gameDate")
        away_rows = history_df[(history_df["AwayTeam"] == team) &
                               (history_df["gameDate"] < game_date)].sort_values("gameDate")
        home_date = home_rows["gameDate"].iloc[-1] if len(home_rows) > 0 else pd.Timestamp.min
        away_date = away_rows["gameDate"].iloc[-1] if len(away_rows) > 0 else pd.Timestamp.min
        if home_date >= away_date and len(home_rows) > 0:
            return home_rows[home_col].iloc[-1]
        elif len(away_rows) > 0:
            return away_rows[away_col].iloc[-1]
        return np.nan

    # Dataset rolling columns
    home_last10_pts_for     = best_last_val(home_team, "Home_last10_pts_for",     "Away_last10_pts_for")
    home_last10_pts_against = best_last_val(home_team, "Home_last10_pts_against", "Away_last10_pts_against")
    away_last10_pts_for     = best_last_val(away_team, "Away_last10_pts_for",     "Home_last10_pts_for")
    away_last10_pts_against = best_last_val(away_team, "Away_last10_pts_against", "Home_last10_pts_against")
    home_last5_pts_for      = best_last_val(home_team, "Home_last5_pts_for",      "Away_last5_pts_for")
    home_last5_pts_against  = best_last_val(home_team, "Home_last5_pts_against",  "Away_last5_pts_against")
    away_last5_pts_for      = best_last_val(away_team, "Away_last5_pts_for",      "Home_last5_pts_for")

    avg_last10_pace = ((home_last10_pts_for or 0) + (home_last10_pts_against or 0) +
                       (away_last10_pts_for or 0) + (away_last10_pts_against or 0))
    opp_adj_total   = ((home_last10_pts_for or 0) + (away_last10_pts_against or 0) +
                       (away_last10_pts_for or 0) + (home_last10_pts_against or 0))

    # Back-to-back: check if team played yesterday
    yesterday = game_date - pd.Timedelta(days=1)
    home_b2b = int(
        ((history_df["HomeTeam"] == home_team) & (history_df["gameDate"] == yesterday)).any() or
        ((history_df["AwayTeam"] == home_team) & (history_df["gameDate"] == yesterday)).any()
    )
    away_b2b = int(
        ((history_df["HomeTeam"] == away_team) & (history_df["gameDate"] == yesterday)).any() or
        ((history_df["AwayTeam"] == away_team) & (history_df["gameDate"] == yesterday)).any()
    )
    both_b2b = int(home_b2b == 1 and away_b2b == 1)

    # Win percentage (season 2025-26 only)
    season_start = pd.Timestamp("2025-10-01")
    def winpct(team):
        home_games = history_df[(history_df["HomeTeam"] == team) &
                                (history_df["gameDate"] >= season_start) &
                                (history_df["gameDate"] < game_date)]
        away_games = history_df[(history_df["AwayTeam"] == team) &
                                (history_df["gameDate"] >= season_start) &
                                (history_df["gameDate"] < game_date)]
        wins = ((home_games["HomePTS"] > home_games["AwayPTS"]).sum() +
                (away_games["AwayPTS"] > away_games["HomePTS"]).sum())
        total_games = len(home_games) + len(away_games)
        return wins / total_games if total_games > 0 else 0.5
    home_winpct_val = winpct(home_team)

    # Head-to-head Total history for this specific pair
    pair = tuple(sorted([home_team, away_team]))
    h2h_games = history_df[
        (history_df.apply(lambda r: tuple(sorted([r["HomeTeam"], r["AwayTeam"]])), axis=1) == pair) &
        (history_df["gameDate"] < game_date)
    ].sort_values("gameDate").tail(5)
    h2h_val = h2h_games["Total"].mean() if len(h2h_games) > 0 else TRAIN_MEAN_TOTAL

    feature_vector = np.array([[
        combined_roll20_total,   # combined_roll20_total
        combined_ewm20_total,    # combined_ewm20_total
        combined_ewm20_poss,     # combined_ewm20_poss
        avg_last10_pace,         # avg_last10_pace
        away_last5_pts_for,      # Away_last5_pts_for
        combined_ewm10_poss,     # combined_ewm10_poss
        h2h_val,                 # h2h_avg_total
        home_last10_pts_against, # Home_last10_pts_against
        home_b2b,                # Home_b2b
        home_winpct_val,         # home_winpct
        both_b2b,                # both_b2b
        home_last5_pts_against,  # Home_last5_pts_against
        opp_adj_total,           # opp_adj_total
    ]], dtype=float)

    # Fill any remaining NaNs with training column medians
    col_medians = np.nanmedian(X_train, axis=0)
    nan_mask = np.isnan(feature_vector)
    feature_vector[nan_mask] = col_medians[np.where(nan_mask)[1]]

    return feature_vector


# Build initial team log (will grow game by game as we predict)
current_team_log = tdf[["gameDate","team","total","poss"]].copy()
current_history  = df.copy()

# Predict game by game in chronological order
print(f"\n  Predicting {len(pred_df)} games for Total...")
total_preds = []

pred_df_sorted = pred_df.sort_values("PredDate").reset_index(drop=True)

for _, game in pred_df_sorted.iterrows():
    game_date  = game["PredDate"]
    home_team  = game["HomeShort"]
    away_team  = game["AwayShort"]

    feats = get_total_features_for_game(
        game_date, home_team, away_team,
        current_history, current_team_log
    )

    feats_sc = scaler_total.transform(feats)
    pred_total = model_total.predict(feats_sc)[0]
    total_preds.append(round(pred_total, 1))

pred_df_sorted["Total_pred"] = total_preds

# Merge predictions back into original order
pred_df = pred_df.merge(
    pred_df_sorted[["PredDate","HomeShort","AwayShort","Total_pred"]],
    left_on=["PredDate","HomeShort","AwayShort"],
    right_on=["PredDate","HomeShort","AwayShort"],
    how="left"
)
pred_df["Total"] = pred_df["Total_pred"]

print(f"  Total predictions: min={pred_df['Total'].min():.1f}  "
      f"max={pred_df['Total'].max():.1f}  mean={pred_df['Total'].mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — SPREAD PREDICTION MODEL  (ADD YOUR CODE HERE)
# ─────────────────────────────────────────────────────────────────────────────
# Spread = Home Points - Away Points
#
# Follow the same pattern as Section A:
#   1. Define your features (SPREAD_FEATURES list)
#   2. Prepare X_train and y_train where y_train = df["Spread"] or df["HomePTS"] - df["AwayPTS"]
#   3. Fit your model (e.g., Ridge, GradientBoosting, etc.)
#   4. For each game in pred_df_sorted, compute features and call model.predict()
#   5. Store results in pred_df["Spread"]
#
# Example skeleton:
#
# SPREAD_FEATURES = [
#     "Home_last10_pts_for", "Home_last10_pts_against",
#     "Away_last10_pts_for", "Away_last10_pts_against",
#     "home_winpct", "away_winpct", "Home_b2b", "Away_b2b",
#     # ... add more features
# ]
#
# train_spread = df.dropna(subset=SPREAD_FEATURES + ["Spread"]).copy()
# X_train_sp = train_spread[SPREAD_FEATURES].values.astype(float)
# y_train_sp = train_spread["Spread"].values
#
# scaler_spread = StandardScaler()
# X_train_sp_sc = scaler_spread.fit_transform(X_train_sp)
#
# model_spread = Ridge(alpha=???)   # tune alpha via TimeSeriesSplit CV
# model_spread.fit(X_train_sp_sc, y_train_sp)
#
# spread_preds = []
# for _, game in pred_df_sorted.iterrows():
#     feats = ... # compute features for this game
#     feats_sc = scaler_spread.transform(feats)
#     spread_preds.append(round(model_spread.predict(feats_sc)[0], 1))
#
# pred_df_sorted["Spread_pred"] = spread_preds
# pred_df = pred_df.merge(pred_df_sorted[["PredDate","HomeShort","AwayShort","Spread_pred"]], ...)
# pred_df["Spread"] = pred_df["Spread_pred"]

print("\n  Spread predictions: NOT YET FILLED (add model in SECTION B)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — OREB PREDICTION MODEL  (ADD YOUR CODE HERE)
# ─────────────────────────────────────────────────────────────────────────────
# OREB = Total Offensive Rebounds (Home + Away)
#
# Follow the same pattern as Section A:
#   1. Define your features (OREB_FEATURES list)
#   2. Train on df["TotalOREB"] as the target
#   3. For each future game, compute features and predict
#   4. Store results in pred_df["OREB"]
#
# Useful columns in the dataset for OREB models:
#   Home_offensiveRebounds, Away_offensiveRebounds  (current game — leakage!)
#   Home_last5_oreb, Home_last10_oreb               (if pre-computed in dataset)
#   Home_last5_pts_for, Away_last5_pts_for           (pace as OREB proxy)
#
# Same warning as Total: watch for leakage. Any column computed from the
# current game's box score (e.g. Home_offensiveRebounds) cannot be used.
#
# Example skeleton:
#
# OREB_FEATURES = [
#     # ... your features here
# ]
#
# train_oreb = df.dropna(subset=OREB_FEATURES + ["TotalOREB"]).copy()
# X_train_or = train_oreb[OREB_FEATURES].values.astype(float)
# y_train_or = train_oreb["TotalOREB"].values
#
# scaler_oreb = StandardScaler()
# X_train_or_sc = scaler_oreb.fit_transform(X_train_or)
#
# model_oreb = Ridge(alpha=???)
# model_oreb.fit(X_train_or_sc, y_train_or)
#
# oreb_preds = []
# for _, game in pred_df_sorted.iterrows():
#     feats = ...
#     feats_sc = scaler_oreb.transform(feats)
#     oreb_preds.append(round(model_oreb.predict(feats_sc)[0], 1))
#
# pred_df["OREB"] = oreb_preds

print("  OREB predictions:  NOT YET FILLED (add model in SECTION C)")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

output_cols = ["Date", "Home", "Away", "Spread", "Total", "OREB"]

# Remove commas from Date (e.g. "Sat, Mar 14, 2026" -> "Sat Mar 14 2026")
# so Excel opens the CSV cleanly without misreading columns
# pred_df["Date"] = (pred_df["Date"]
#                    .str.replace(",", "", regex=False)
#                    .str.replace(r"\s+", " ", regex=True)
#                    .str.strip())

# utf-8-sig adds the BOM so Excel detects encoding correctly
pred_df[output_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n{'='*60}")
print(f"Saved: {OUTPUT_CSV}")
print(f"  Rows: {len(pred_df)}")
print(f"  Total filled:  {pred_df['Total'].notna().sum()}")
print(f"  Spread filled: {pred_df['Spread'].notna().sum()}")
print(f"  OREB filled:   {pred_df['OREB'].notna().sum()}")
print(f"{'='*60}")