from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from xgboost import XGBRegressor

DATA_PATH = Path(__file__).resolve().parent / "Data" / "final" / "MasterDataset.csv"
START_DATE = pd.Timestamp("2021-10-19")
TEST_START_DATE = pd.Timestamp("2026-01-14")
TARGET = "Spread"
ROLLING_WINDOW = 20
SEASON_START_MONTH = 9
CV_FOLDS = 5

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
    "Home_poss",
    "Away_poss",
    "Home_rest_days",
    "Away_rest_days",
    "Spread",
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

# Selected XGBoost settings from the broader sweep.
XGBOOST_PARAMS = {
    "n_estimators": 160,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 25,
    "subsample": 0.55,
    "colsample_bytree": 0.55,
    "reg_lambda": 10.0,
    "reg_alpha": 2.0,
    "gamma": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_GRID = {
    "n_estimators": [160],
    "learning_rate": [0.03],
    "max_depth": [2],
    "min_child_weight": [25],
    "subsample": [0.55],
    "colsample_bytree": [0.55],
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
            "pts_for": home_pts,
            "pts_against": away_pts,
            "poss": pd.to_numeric(df["Home_poss"], errors="coerce"),
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
            "pts_for": away_pts,
            "pts_against": home_pts,
            "poss": pd.to_numeric(df["Away_poss"], errors="coerce"),
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
    prior_pts_for = grouped["pts_for"].cumsum() - season_df["pts_for"]
    prior_pts_against = grouped["pts_against"].cumsum() - season_df["pts_against"]
    prior_poss = grouped["poss"].cumsum() - season_df["poss"]
    season_df["off_rating_calc"] = 100.0 * prior_pts_for / prior_poss.replace(0.0, np.nan)
    season_df["def_rating_calc"] = 100.0 * prior_pts_against / prior_poss.replace(0.0, np.nan)
    season_df["net_rating_calc"] = season_df["off_rating_calc"] - season_df["def_rating_calc"]

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
        [
            "gameDate",
            "season",
            "teamId",
            "winpct_calc",
            "home_split_margin_calc",
            "off_rating_calc",
            "def_rating_calc",
            "net_rating_calc",
        ],
    ].rename(
        columns={
            "teamId": "HomeTeamId",
            "winpct_calc": "home_winpct_calc",
            "off_rating_calc": "home_off_rating_calc",
            "def_rating_calc": "home_def_rating_calc",
            "net_rating_calc": "home_net_rating_calc",
        }
    )
    away_stats = season_df.loc[
        season_df["is_home"] == 0.0,
        [
            "gameDate",
            "season",
            "teamId",
            "winpct_calc",
            "road_split_margin_calc",
            "off_rating_calc",
            "def_rating_calc",
            "net_rating_calc",
        ],
    ].rename(
        columns={
            "teamId": "AwayTeamId",
            "winpct_calc": "away_winpct_calc",
            "road_split_margin_calc": "away_road_split_margin_calc",
            "off_rating_calc": "away_off_rating_calc",
            "def_rating_calc": "away_def_rating_calc",
            "net_rating_calc": "away_net_rating_calc",
        }
    )

    merged = df.merge(home_stats, on=["gameDate", "season", "HomeTeamId"], how="left")
    merged = merged.merge(away_stats, on=["gameDate", "season", "AwayTeamId"], how="left")
    merged["winpct_diff_calc"] = merged["home_winpct_calc"] - merged["away_winpct_calc"]
    merged["split_margin_diff_calc"] = merged["home_split_margin_calc"] - merged["away_road_split_margin_calc"]
    merged["offRating_diff"] = merged["home_off_rating_calc"] - merged["away_off_rating_calc"]
    merged["defRating_diff"] = merged["home_def_rating_calc"] - merged["away_def_rating_calc"]
    merged["netRating_diff"] = merged["home_net_rating_calc"] - merged["away_net_rating_calc"]
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
    train_df = df[df["gameDate"] < TEST_START_DATE].copy()
    test_df = df[df["gameDate"] >= TEST_START_DATE].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Date-based split produced an empty train or test set.")
    return train_df, test_df


def prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Tree models do not need scaling; only leak-safe train-median imputation.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.astype(float).fillna(medians).fillna(0.0)
    X_test = X_test.astype(float).fillna(medians).fillna(0.0)
    return X_train, X_test


def make_cv_splits(train_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    game_days = train_df["gameDate"].dt.normalize()
    unique_days = pd.Index(game_days.drop_duplicates().sort_values())
    splitter = TimeSeriesSplit(n_splits=CV_FOLDS)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for day_train_idx, day_valid_idx in splitter.split(unique_days):
        train_days = unique_days[day_train_idx]
        valid_days = unique_days[day_valid_idx]
        row_train_idx = train_df.index[game_days.isin(train_days)].to_numpy()
        row_valid_idx = train_df.index[game_days.isin(valid_days)].to_numpy()
        splits.append((row_train_idx, row_valid_idx))

    return splits


def build_history_lookup(history: pd.DataFrame) -> dict[float, pd.DataFrame]:
    return {
        team_id: group.reset_index(drop=True)
        for team_id, group in history.groupby("teamId", sort=False)
    }


def get_team_history(
    history_by_team: dict[float, pd.DataFrame],
    team_id: float,
    game_date: pd.Timestamp,
    season: int | None = None,
) -> pd.DataFrame:
    history = history_by_team.get(team_id)
    if history is None:
        return pd.DataFrame()

    past = history[history["gameDate"] < game_date]
    if season is not None:
        past = past[past["season"] == season]
    return past


def rolling_team_value(
    history_by_team: dict[float, pd.DataFrame],
    team_id: float,
    game_date: pd.Timestamp,
    column: str,
    window: int,
) -> float:
    past = get_team_history(history_by_team, team_id, game_date)
    if len(past) < 3:
        return np.nan
    return float(past[column].tail(window).mean())


def season_team_values(
    history_by_team: dict[float, pd.DataFrame],
    team_id: float,
    game_date: pd.Timestamp,
) -> dict[str, float]:
    season = season_key(pd.Series([game_date])).iloc[0]
    past = get_team_history(history_by_team, team_id, game_date, season=season)
    if past.empty:
        return {
            "winpct": np.nan,
            "home_split_margin": np.nan,
            "road_split_margin": np.nan,
            "net_rating": np.nan,
            "def_rating": np.nan,
        }

    rating_rows = past[["pts_for", "pts_against", "poss"]].dropna()
    poss_sum = rating_rows["poss"].sum()
    if poss_sum > 0:
        net_rating = 100.0 * (rating_rows["pts_for"].sum() - rating_rows["pts_against"].sum()) / poss_sum
        def_rating = 100.0 * rating_rows["pts_against"].sum() / poss_sum
    else:
        net_rating = np.nan
        def_rating = np.nan

    return {
        "winpct": float(past["win"].mean()),
        "home_split_margin": float(past.loc[past["is_home"] == 1.0, "margin"].mean()),
        "road_split_margin": float(past.loc[past["is_home"] == 0.0, "margin"].mean()),
        "net_rating": float(net_rating),
        "def_rating": float(def_rating),
    }


def h2h_margin_from_history(
    history_df: pd.DataFrame,
    home_team_id: float,
    away_team_id: float,
    game_date: pd.Timestamp,
) -> float:
    pair_games = history_df[
        (
            ((history_df["HomeTeamId"] == home_team_id) & (history_df["AwayTeamId"] == away_team_id))
            | ((history_df["HomeTeamId"] == away_team_id) & (history_df["AwayTeamId"] == home_team_id))
        )
        & (history_df["gameDate"] < game_date)
    ].sort_values("gameDate")
    if pair_games.empty:
        return np.nan

    spreads = pd.to_numeric(pair_games.tail(5)[TARGET], errors="coerce")
    oriented = np.where(pair_games.tail(5)["HomeTeamId"] == home_team_id, spreads, -spreads)
    return float(np.nanmean(oriented))


def build_snapshot_test_frame(history_source: pd.DataFrame, test_source: pd.DataFrame) -> pd.DataFrame:
    history = build_team_history(history_source)
    history_by_team = build_history_lookup(history)
    rows: list[dict[str, float | pd.Timestamp]] = []

    for row in test_source.itertuples(index=False):
        home_team_id = float(row.HomeTeamId)
        away_team_id = float(row.AwayTeamId)
        game_date = pd.Timestamp(row.gameDate)
        home_season = season_team_values(history_by_team, home_team_id, game_date)
        away_season = season_team_values(history_by_team, away_team_id, game_date)

        rows.append(
            {
                "gameDate": game_date,
                "home_indicator": 1.0,
                "diff_last10_margin": rolling_team_value(history_by_team, home_team_id, game_date, "margin", 10)
                - rolling_team_value(history_by_team, away_team_id, game_date, "margin", 10),
                "netrating_diff": home_season["net_rating"] - away_season["net_rating"],
                "split_margin_diff_calc": home_season["home_split_margin"] - away_season["road_split_margin"],
                "away_rest_days_capped": float(np.clip(pd.to_numeric(row.Away_rest_days, errors="coerce"), 0, 4)),
                "home_rest_days_capped": float(np.clip(pd.to_numeric(row.Home_rest_days, errors="coerce"), 0, 4)),
                "diff_roll20_tov_rate": rolling_team_value(history_by_team, home_team_id, game_date, "tov_rate", 20)
                - rolling_team_value(history_by_team, away_team_id, game_date, "tov_rate", 20),
                "Home_b2b": float(pd.to_numeric(row.Home_b2b, errors="coerce")),
                "defRating_diff": home_season["def_rating"] - away_season["def_rating"],
                "winpct_diff_calc": home_season["winpct"] - away_season["winpct"],
                "diff_roll20_ts": rolling_team_value(history_by_team, home_team_id, game_date, "ts", 20)
                - rolling_team_value(history_by_team, away_team_id, game_date, "ts", 20),
                "diff_roll20_3pa_rate": rolling_team_value(history_by_team, home_team_id, game_date, "three_pa_rate", 20)
                - rolling_team_value(history_by_team, away_team_id, game_date, "three_pa_rate", 20),
                "h2h_margin_for_home": h2h_margin_from_history(history_source, home_team_id, away_team_id, game_date),
                "Away_b2b": float(pd.to_numeric(row.Away_b2b, errors="coerce")),
                TARGET: float(pd.to_numeric(row.Spread, errors="coerce")),
            }
        )

    model_df = pd.DataFrame(rows)
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    return model_df.dropna(subset=[TARGET]).reset_index(drop=True)


def cv_mae_for_params(train_df: pd.DataFrame, params: dict[str, float | int]) -> list[float]:
    fold_maes: list[float] = []

    for row_train_idx, row_valid_idx in make_cv_splits(train_df):
        fold_train = train_df.loc[row_train_idx]
        fold_valid = train_df.loc[row_valid_idx]

        X_fold_train = fold_train[FEATURES].copy()
        X_fold_valid = fold_valid[FEATURES].copy()
        y_fold_train = fold_train[TARGET].copy()
        y_fold_valid = fold_valid[TARGET].copy()

        X_fold_train, X_fold_valid = prepare_features(X_fold_train, X_fold_valid)
        model = XGBRegressor(**(XGBOOST_PARAMS | params))
        model.fit(X_fold_train, y_fold_train)
        fold_pred = model.predict(X_fold_valid)
        fold_maes.append(mean_absolute_error(y_fold_valid, fold_pred))

    return fold_maes


def tune_xgboost(train_df: pd.DataFrame) -> tuple[dict[str, float | int], list[float], float]:
    best_params = XGBOOST_PARAMS.copy()
    best_fold_maes: list[float] = []
    best_cv_mae = float("inf")

    for params in ParameterGrid(XGBOOST_GRID):
        fold_maes = cv_mae_for_params(train_df, params)
        cv_mae = float(np.mean(fold_maes))
        if cv_mae < best_cv_mae:
            best_params = XGBOOST_PARAMS | params
            best_fold_maes = fold_maes
            best_cv_mae = cv_mae

    return best_params, best_fold_maes, best_cv_mae


def main() -> None:
    raw_df = load_data()
    raw_train_df, raw_test_df = split_data(raw_df)
    train_df = build_model_frame(raw_train_df)
    test_df = build_snapshot_test_frame(raw_train_df, raw_test_df)
    model_df = pd.concat([train_df, test_df], ignore_index=True)

    best_params, fold_maes, cv_mae = tune_xgboost(train_df)

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()
    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    X_train, X_test = prepare_features(X_train, X_test)
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    importances = pd.Series(model.feature_importances_, index=FEATURES, dtype=float).sort_values(ascending=False)

    print(f"Rows: {len(model_df)}  |  {model_df['gameDate'].min()} -> {model_df['gameDate'].max()}")
    print(f"Start date filter: {START_DATE.date()}")
    print(f"Date-based split: {len(train_df)} train / {len(test_df)} test")
    print(f"Train end: {raw_train_df['gameDate'].iloc[-1]}")
    print(f"Test start: {raw_test_df['gameDate'].iloc[0]}")
    print(f"Held-out test window: {TEST_START_DATE.date()} -> {model_df['gameDate'].max().date()}")
    print("Held-out test features frozen to information available before 2026-01-14")
    print(f"5-fold time-series CV on the pre-{TEST_START_DATE.date()} training block")
    print(f"Rolling window: {ROLLING_WINDOW}")
    print()
    print(f"Best params: {best_params}")
    print(f"Fold MAE: {[round(mae, 4) for mae in fold_maes]}")
    print(f"CV MAE: {cv_mae:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print()
    print("Feature importance:")
    print(importances.round(4).to_string())


if __name__ == "__main__":
    main()
