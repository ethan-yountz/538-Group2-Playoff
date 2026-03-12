from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

DATA_PATH = Path(__file__).resolve().parent / "Data" / "final" / "MasterDataset.csv"
START_DATE = pd.Timestamp("2021-10-19")
TEST_FRACTION = 0.20
TARGET = "Spread"
ROLLING_WINDOW = 20
SEASON_START_MONTH = 9

FEATURES = [
    "home_indicator",
    "diff_last10_margin",
    "netrating_diff",
    "split_margin_diff_calc",
    "away_rest_days_capped",
    "home_rest_days_capped",
    "diff_roll20_tov_rate",
    "Home_b2b",
    "defRating_diff",
    "winpct_diff_calc",
    "diff_roll20_ts",
    "diff_roll20_3pa_rate",
    "h2h_margin_for_home",
    "Away_b2b",
]

REQUIRED_COLUMNS = [
    "gameDate",
    "HomeTeamId",
    "AwayTeamId",
    "HomePTS",
    "AwayPTS",
    "Home_rest_days",
    "Away_rest_days",
    "Spread",
    "netRating_diff",
    "defRating_diff",
    "diff_last10_margin",
    "Home_b2b",
    "Away_b2b",
    "Home_TS",
    "Away_TS",
    "Home_3PA_rate",
    "Away_3PA_rate",
    "Home_TOV_rate",
    "Away_TOV_rate",
]

# Best single-holdout XGBoost settings from the sweep.
XGBOOST_PARAMS = {
    "n_estimators": 160,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_child_weight": 25,
    "subsample": 0.65,
    "colsample_bytree": 0.70,
    "reg_lambda": 10.0,
    "reg_alpha": 2.0,
    "gamma": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "random_state": 42,
    "n_jobs": -1,
}


def season_key(game_dates: pd.Series) -> pd.Series:
    years = game_dates.dt.year
    return years.where(game_dates.dt.month >= SEASON_START_MONTH, years - 1)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    df = df.dropna(subset=["gameDate"]).sort_values("gameDate").reset_index(drop=True)
    df = df[df["gameDate"] >= START_DATE].copy()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"MasterDataset is missing required columns: {missing}")

    df["season"] = season_key(df["gameDate"])
    df["HomeTeamId"] = pd.to_numeric(df["HomeTeamId"], errors="coerce")
    df["AwayTeamId"] = pd.to_numeric(df["AwayTeamId"], errors="coerce")
    return df


def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    home_pts = pd.to_numeric(df["HomePTS"], errors="coerce")
    away_pts = pd.to_numeric(df["AwayPTS"], errors="coerce")
    margin = home_pts - away_pts

    # One team-by-game log keeps the rolling features venue-agnostic.
    home_rows = pd.DataFrame(
        {
            "gameDate": df["gameDate"],
            "season": df["season"],
            "teamId": df["HomeTeamId"],
            "is_home": 1.0,
            "margin": margin,
            "win": (margin > 0).astype(float),
            "ts": pd.to_numeric(df["Home_TS"], errors="coerce"),
            "three_pa_rate": pd.to_numeric(df["Home_3PA_rate"], errors="coerce"),
            "tov_rate": pd.to_numeric(df["Home_TOV_rate"], errors="coerce"),
        }
    )
    away_rows = pd.DataFrame(
        {
            "gameDate": df["gameDate"],
            "season": df["season"],
            "teamId": df["AwayTeamId"],
            "is_home": 0.0,
            "margin": -margin,
            "win": (margin < 0).astype(float),
            "ts": pd.to_numeric(df["Away_TS"], errors="coerce"),
            "three_pa_rate": pd.to_numeric(df["Away_3PA_rate"], errors="coerce"),
            "tov_rate": pd.to_numeric(df["Away_TOV_rate"], errors="coerce"),
        }
    )

    history = pd.concat([home_rows, away_rows], ignore_index=True)
    return history.sort_values(["teamId", "gameDate", "is_home"]).reset_index(drop=True)


def add_season_features(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    season_df = history.copy()
    grouped = season_df.groupby(["teamId", "season"], sort=False)

    prior_games = grouped.cumcount().astype(float)
    season_df["winpct_calc"] = (grouped["win"].cumsum() - season_df["win"]) / prior_games.replace(0.0, np.nan)

    season_df["home_game"] = season_df["is_home"]
    season_df["road_game"] = 1.0 - season_df["is_home"]
    season_df["home_margin_value"] = np.where(season_df["is_home"] == 1.0, season_df["margin"], 0.0)
    season_df["road_margin_value"] = np.where(season_df["is_home"] == 0.0, season_df["margin"], 0.0)

    prior_home_games = grouped["home_game"].cumsum() - season_df["home_game"]
    prior_road_games = grouped["road_game"].cumsum() - season_df["road_game"]
    prior_home_margin = grouped["home_margin_value"].cumsum() - season_df["home_margin_value"]
    prior_road_margin = grouped["road_margin_value"].cumsum() - season_df["road_margin_value"]

    season_df["home_split_margin_calc"] = prior_home_margin / prior_home_games.replace(0.0, np.nan)
    season_df["road_split_margin_calc"] = prior_road_margin / prior_road_games.replace(0.0, np.nan)

    home_stats = season_df.loc[
        season_df["is_home"] == 1.0,
        ["gameDate", "season", "teamId", "winpct_calc", "home_split_margin_calc"],
    ].rename(
        columns={
            "teamId": "HomeTeamId",
            "winpct_calc": "home_winpct_calc",
        }
    )
    away_stats = season_df.loc[
        season_df["is_home"] == 0.0,
        ["gameDate", "season", "teamId", "winpct_calc", "road_split_margin_calc"],
    ].rename(
        columns={
            "teamId": "AwayTeamId",
            "winpct_calc": "away_winpct_calc",
            "road_split_margin_calc": "away_road_split_margin_calc",
        }
    )

    merged = df.merge(home_stats, on=["gameDate", "season", "HomeTeamId"], how="left")
    merged = merged.merge(away_stats, on=["gameDate", "season", "AwayTeamId"], how="left")
    merged["winpct_diff_calc"] = merged["home_winpct_calc"] - merged["away_winpct_calc"]
    merged["split_margin_diff_calc"] = merged["home_split_margin_calc"] - merged["away_road_split_margin_calc"]
    return merged


def add_h2h_feature(df: pd.DataFrame) -> pd.DataFrame:
    merged = df.copy()
    spread = pd.to_numeric(merged[TARGET], errors="coerce")

    merged["pair_low"] = np.minimum(merged["HomeTeamId"], merged["AwayTeamId"])
    merged["pair_high"] = np.maximum(merged["HomeTeamId"], merged["AwayTeamId"])
    merged["pair_margin_canonical"] = np.where(merged["HomeTeamId"] < merged["AwayTeamId"], spread, -spread)
    merged["pair_margin_roll"] = (
        merged.groupby(["pair_low", "pair_high"], sort=False)["pair_margin_canonical"]
        .transform(lambda series: series.shift(1).rolling(5, min_periods=1).mean())
    )
    merged["h2h_margin_for_home"] = np.where(
        merged["HomeTeamId"] < merged["AwayTeamId"],
        merged["pair_margin_roll"],
        -merged["pair_margin_roll"],
    )
    return merged


def add_rolling_features(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    rolling = history.copy()
    grouped = rolling.groupby("teamId", sort=False)

    # shift(1) keeps the current game out of its own rolling features.
    rolling["roll20_ts"] = grouped["ts"].transform(
        lambda series: series.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )
    rolling["roll20_3pa_rate"] = grouped["three_pa_rate"].transform(
        lambda series: series.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )
    rolling["roll20_tov_rate"] = grouped["tov_rate"].transform(
        lambda series: series.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )

    home_rolls = rolling.loc[
        rolling["is_home"] == 1.0,
        ["gameDate", "season", "teamId", "roll20_ts", "roll20_3pa_rate", "roll20_tov_rate"],
    ].rename(
        columns={
            "teamId": "HomeTeamId",
            "roll20_ts": "home_roll20_ts",
            "roll20_3pa_rate": "home_roll20_3pa_rate",
            "roll20_tov_rate": "home_roll20_tov_rate",
        }
    )
    away_rolls = rolling.loc[
        rolling["is_home"] == 0.0,
        ["gameDate", "season", "teamId", "roll20_ts", "roll20_3pa_rate", "roll20_tov_rate"],
    ].rename(
        columns={
            "teamId": "AwayTeamId",
            "roll20_ts": "away_roll20_ts",
            "roll20_3pa_rate": "away_roll20_3pa_rate",
            "roll20_tov_rate": "away_roll20_tov_rate",
        }
    )

    merged = df.merge(home_rolls, on=["gameDate", "season", "HomeTeamId"], how="left")
    merged = merged.merge(away_rolls, on=["gameDate", "season", "AwayTeamId"], how="left")
    merged["diff_roll20_ts"] = merged["home_roll20_ts"] - merged["away_roll20_ts"]
    merged["diff_roll20_3pa_rate"] = merged["home_roll20_3pa_rate"] - merged["away_roll20_3pa_rate"]
    merged["diff_roll20_tov_rate"] = merged["home_roll20_tov_rate"] - merged["away_roll20_tov_rate"]
    return merged


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["home_rest_days_capped"] = pd.to_numeric(working["Home_rest_days"], errors="coerce").clip(0, 4)
    working["away_rest_days_capped"] = pd.to_numeric(working["Away_rest_days"], errors="coerce").clip(0, 4)

    history = build_team_history(working)
    working = add_season_features(working, history)
    working = add_h2h_feature(working)
    working = add_rolling_features(working, history)

    model_df = pd.DataFrame(
        {
            "gameDate": working["gameDate"],
            "home_indicator": 1.0,
            "diff_last10_margin": pd.to_numeric(working["diff_last10_margin"], errors="coerce"),
            "netrating_diff": pd.to_numeric(working["netRating_diff"], errors="coerce"),
            "split_margin_diff_calc": pd.to_numeric(working["split_margin_diff_calc"], errors="coerce"),
            "away_rest_days_capped": pd.to_numeric(working["away_rest_days_capped"], errors="coerce"),
            "home_rest_days_capped": pd.to_numeric(working["home_rest_days_capped"], errors="coerce"),
            "diff_roll20_tov_rate": pd.to_numeric(working["diff_roll20_tov_rate"], errors="coerce"),
            "Home_b2b": pd.to_numeric(working["Home_b2b"], errors="coerce"),
            "defRating_diff": pd.to_numeric(working["defRating_diff"], errors="coerce"),
            "winpct_diff_calc": pd.to_numeric(working["winpct_diff_calc"], errors="coerce"),
            "diff_roll20_ts": pd.to_numeric(working["diff_roll20_ts"], errors="coerce"),
            "diff_roll20_3pa_rate": pd.to_numeric(working["diff_roll20_3pa_rate"], errors="coerce"),
            "h2h_margin_for_home": pd.to_numeric(working["h2h_margin_for_home"], errors="coerce"),
            "Away_b2b": pd.to_numeric(working["Away_b2b"], errors="coerce"),
            TARGET: pd.to_numeric(working[TARGET], errors="coerce"),
        }
    )

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    return model_df.dropna(subset=[TARGET]).reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Tree models do not need scaling; only leak-safe train-median imputation.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.astype(float).fillna(medians).fillna(0.0)
    X_test = X_test.astype(float).fillna(medians).fillna(0.0)
    return X_train, X_test


def main() -> None:
    model_df = build_model_frame(load_data())
    train_df, test_df = split_data(model_df)

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()
    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    X_train, X_test = prepare_features(X_train, X_test)
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    importances = pd.Series(model.feature_importances_, index=FEATURES, dtype=float).sort_values(ascending=False)

    print(f"Rows: {len(model_df)}  |  {model_df['gameDate'].min()} -> {model_df['gameDate'].max()}")
    print(f"Start date filter: {START_DATE.date()}")
    print(f"Chronological split: {len(train_df)} train / {len(test_df)} test")
    print(f"Train end: {train_df['gameDate'].iloc[-1]}")
    print(f"Test start: {test_df['gameDate'].iloc[0]}")
    print(f"Rolling window: {ROLLING_WINDOW}")
    print()
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print()
    print("Feature importance:")
    print(importances.round(4).to_string())


if __name__ == "__main__":
    main()
