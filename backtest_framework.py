from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def _ensure_targets(df: pd.DataFrame, team_stats_path: Path) -> pd.DataFrame:
    # Build the 3 targets needed for the assignment if they are not already present.
    if "spread" not in df.columns:
        df["spread"] = df["homeScore"] - df["awayScore"]
    if "total" not in df.columns:
        df["total"] = df["homeScore"] + df["awayScore"]
    if "oreb" in df.columns:
        return df

    # Prefer already-merged offensive rebound columns if available.
    if {"homeReboundsOffensive", "awayReboundsOffensive"} <= set(df.columns):
        df["oreb"] = pd.to_numeric(df["homeReboundsOffensive"], errors="coerce") + pd.to_numeric(
            df["awayReboundsOffensive"], errors="coerce"
        )
        return df

    # Otherwise, pull rebounds from TeamStatistics game rows and merge onto each game.
    stats = pd.read_csv(team_stats_path, usecols=["gameId", "teamId", "home", "reboundsOffensive"])
    stats["gameId"] = stats["gameId"].astype(str)
    stats["teamId"] = stats["teamId"].astype(str)

    home_id = "hometeamId" if "hometeamId" in df.columns else "homeTeamId"
    away_id = "awayteamId" if "awayteamId" in df.columns else "awayTeamId"
    df["gameId"] = df["gameId"].astype(str)
    df[home_id] = df[home_id].astype(str)
    df[away_id] = df[away_id].astype(str)

    home = stats.query("home == 1")[["gameId", "teamId", "reboundsOffensive"]].rename(
        columns={"teamId": home_id, "reboundsOffensive": "homeReboundsOffensive"}
    )
    away = stats.query("home == 0")[["gameId", "teamId", "reboundsOffensive"]].rename(
        columns={"teamId": away_id, "reboundsOffensive": "awayReboundsOffensive"}
    )
    df = df.merge(home, on=["gameId", home_id], how="left")
    df = df.merge(away, on=["gameId", away_id], how="left")
    df["oreb"] = pd.to_numeric(df["homeReboundsOffensive"], errors="coerce") + pd.to_numeric(
        df["awayReboundsOffensive"], errors="coerce"
    )
    return df


def _build_naive_pred(target, row, train, home_id, away_id):
    # Naive baseline rules per assignment:
    # - spread: season team margins
    # - total: season offense averages for each side
    # - oreb: constant league baseline (23)
    s = row["season"]
    if target == "spread":
        home_margin = train.groupby([home_id, "season"])["margin"].mean()
        away_margin = (-train["margin"]).groupby([train[away_id], train["season"]]).mean()
        league_margin = train.groupby("season")["margin"].mean()
        hm = home_margin.get((row[home_id], s), league_margin.get(s, np.nan) / 2)
        am = away_margin.get((row[away_id], s), -league_margin.get(s, np.nan) / 2)
        return (hm - am) / 2
    if target == "total":
        home_pts = train.groupby([home_id, "season"])["homeScore"].mean()
        away_pts = train.groupby([away_id, "season"])["awayScore"].mean()
        league_home = train.groupby("season")["homeScore"].mean()
        league_away = train.groupby("season")["awayScore"].mean()
        return home_pts.get((row[home_id], s), league_home.get(s, np.nan)) + away_pts.get(
            (row[away_id], s), league_away.get(s, np.nan)
        )
    return 23.0


def run_backtest(
    data_path: Path = Path(__file__).resolve().parent / "Data" / "Games.csv",
    post_bubble_start: str = "2020-07-30",
    test_fraction: float = 0.20,
    team_stats_path: Path | None = None,
    model: str = "naive",
    feature_cols: list[str] | None = None,
):
    # Feature arg kept for future expansion; currently only naive model is implemented.
    if team_stats_path is None:
        team_stats_path = Path(__file__).resolve().parent / "Data" / "TeamStatistics.csv"

    # Load and time-filter schedule data
    df = pd.read_csv(data_path, low_memory=False)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df = df.dropna(subset=["gameDateTimeEst"])
    df = df[df["gameDateTimeEst"] >= pd.Timestamp(post_bubble_start)].sort_values("gameDateTimeEst").copy()

    home_id = "hometeamId" if "hometeamId" in df.columns else "homeTeamId"
    away_id = "awayteamId" if "awayteamId" in df.columns else "awayTeamId"

    if home_id not in df.columns or away_id not in df.columns:
        raise ValueError("Expected hometeamId/homeTeamId and awayteamId/awayTeamId columns.")

    # Ensure all three target variables exist before modeling.
    df = _ensure_targets(df, team_stats_path)
    df["season"] = df["gameDateTimeEst"].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
    df["margin"] = df["homeScore"] - df["awayScore"]

    # Chronological 80/20 post-bubble split.
    split = int(len(df) * (1 - test_fraction))
    split = max(1, min(split, len(df) - 1))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    results: dict[str, float] = {}
    preds = test[["gameDateTimeEst"]].copy()

    # Only naive model is enabled for now.
    if model != "naive":
        raise ValueError("Only model='naive' is currently supported. Add new models later.")

    # Evaluate each target with its dedicated naive formula.
    for target in ["spread", "total", "oreb"]:
        pred = pd.Series([_build_naive_pred(target, row, train, home_id, away_id) for _, row in test.iterrows()], index=test.index)
        y = pd.to_numeric(test[target], errors="coerce")
        mask = pred.notna() & y.notna()
        # MAE on non-null predictions
        results[target] = float(mean_absolute_error(y[mask], pred[mask])) if mask.any() else np.nan
        preds[f"{target}_pred"] = pred

    # Keep one summary score for quick comparison.
    results["avg_mae"] = float(np.nanmean(list(results.values())))
    return {"split_date": df.iloc[split - 1]["gameDateTimeEst"], "mae": results}, preds


if __name__ == "__main__":
    scores, preds = run_backtest(model="naive")
    print(scores["split_date"], scores["mae"]["avg_mae"])
    for k, v in scores["mae"].items():
        print(f"{k}: {v}")
    print(preds.head())
