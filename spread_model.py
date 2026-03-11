from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = Path(__file__).resolve().parent / "Data" / "final" / "MasterDataset.csv"
START_DATE = pd.Timestamp("2021-10-19")
TEST_FRACTION = 0.20
TARGET = "Spread"

# Pregame inputs used by the stepwise search.
FEATURE_MAP = {
    "elo_diff": "EloDiff",
    "netRating_diff": "netRating_diff",
    "offRating_diff": "offRating_diff",
    "defRating_diff": "defRating_diff",
    "pace_diff": "pace_diff",
    "margin_last5_diff": "diff_last5_margin",
    "rest_days_diff": "rest_diff",
}


def load_data() -> pd.DataFrame:
    # Keep the games in true time order so both the split and CV stay chronological.
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    df = df.dropna(subset=["gameDate"]).sort_values("gameDate").reset_index(drop=True)
    df = df[df["gameDate"] >= START_DATE].copy()

    missing = [column for column in FEATURE_MAP.values() if column not in df.columns]
    if missing:
        raise ValueError(f"MasterDataset is missing required columns: {missing}")

    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy(), split_idx


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Build one clean feature matrix and drop rows without a spread target.
    X = pd.DataFrame({"home_indicator": np.ones(len(df), dtype=float)}, index=df.index)
    for name, source in FEATURE_MAP.items():
        X[name] = pd.to_numeric(df[source], errors="coerce")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    mask = y.notna()
    X = X.loc[mask].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y


def fill_missing(X_train: pd.DataFrame, X_other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Use training medians only so later rows do not leak into earlier rows.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_other = X_other.fillna(medians).fillna(0.0)
    return X_train, X_other


def fit_and_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
) -> tuple[float, float, pd.Series]:
    X_train_use, X_test_use = fill_missing(X_train[features].copy(), X_test[features].copy())

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_use, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train_use))
    test_mae = mean_absolute_error(y_test, model.predict(X_test_use))
    coefs = pd.Series(model.coef_, index=features).sort_values(key=np.abs, ascending=False)
    return float(train_mae), float(test_mae), coefs


def cv_mae(X_train: pd.DataFrame, y_train: pd.Series, features: list[str]) -> float:
    # Train on older games, validate on the next block of newer games.
    n_splits = min(5, max(2, len(X_train) // 250))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    maes = []

    for train_idx, val_idx in splitter.split(X_train):
        X_tr = X_train.iloc[train_idx][features].copy()
        X_val = X_train.iloc[val_idx][features].copy()
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        X_tr, X_val = fill_missing(X_tr, X_val)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_tr, y_tr)
        maes.append(mean_absolute_error(y_val, model.predict(X_val)))

    return float(np.mean(maes))


def stepwise_select(X_train: pd.DataFrame, y_train: pd.Series, min_gain: float = 0.001) -> tuple[list[str], list[tuple[str, str, float]], float]:
    # Start from home court only, then add or remove features if CV MAE improves.
    selected = ["home_indicator"]
    remaining = [feature for feature in X_train.columns if feature not in selected]
    best_cv = cv_mae(X_train, y_train, selected)
    history: list[tuple[str, str, float]] = [("start", "home_indicator", best_cv)]

    while remaining:
        best_feature = None
        best_feature_cv = best_cv

        for feature in remaining:
            score = cv_mae(X_train, y_train, selected + [feature])
            if score < best_feature_cv - min_gain:
                best_feature = feature
                best_feature_cv = score

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_cv = best_feature_cv
        history.append(("add", best_feature, best_cv))

        changed = True
        while changed:
            changed = False
            for feature in selected[1:]:
                trial = [name for name in selected if name != feature]
                score = cv_mae(X_train, y_train, trial)
                if score < best_cv - min_gain:
                    selected.remove(feature)
                    remaining.append(feature)
                    best_cv = score
                    history.append(("remove", feature, best_cv))
                    changed = True
                    break

    return selected, history, best_cv


def main() -> None:
    df = load_data()
    train_df, test_df, split_idx = split_data(df)
    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)

    selected, history, final_cv = stepwise_select(X_train, y_train)
    train_mae, test_mae, coefs = fit_and_score(X_train, y_train, X_test, y_test, selected)

    print(f"Rows: {len(df)}  |  {df['gameDate'].min()} -> {df['gameDate'].max()}")
    print(f"Start date filter: {START_DATE.date()}")
    print(
        f"Chronological split: {len(train_df)} train / {len(test_df)} test "
        f"({len(train_df) / len(df):.1%} / {len(test_df) / len(df):.1%})"
    )
    print(f"Split index: {split_idx}")
    print(f"Train end: {train_df['gameDate'].iloc[-1]}")
    print(f"Test start: {test_df['gameDate'].iloc[0]}")
    print()
    print("Feature pool:")
    for name, source in FEATURE_MAP.items():
        print(f"  {name:<18} <- {source}")
    print("  home_indicator      <- constant 1.0 baseline")
    print()
    print("Selection history:")
    for action, feature, score in history:
        print(f"  {action:<6} {feature:<18} CV MAE: {score:.4f}")
    print()
    print(f"Selected features: {selected}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Final CV MAE: {final_cv:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print()
    print("Coefficients:")
    print(coefs.round(4).to_string())


if __name__ == "__main__":
    main()
