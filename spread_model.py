from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path(__file__).resolve().parent / "Data" / "final" / "MasterDataset.csv"
START_DATE = pd.Timestamp("2021-10-19")
TEST_FRACTION = 0.20
TARGET = "Spread"
HOME_FEATURE = "home_indicator"
LASSO_ALPHA = 0.05
MAX_ITER = 20000

FEATURE_MAP = {
    "elo_diff": "EloDiff",
    "netrating_diff": "netRating_diff",
    "offRating_diff": "offRating_diff",
    "defRating_diff": "defRating_diff",
    "pace_diff": "pace_diff",
    "diff_last5_margin": "diff_last5_margin",
    "rest_diff": "rest_diff",
}
FEATURES = [*FEATURE_MAP.keys(), HOME_FEATURE]
SCALED_FEATURES = [feature for feature in FEATURES if feature != HOME_FEATURE]


def load_model_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    df = df.dropna(subset=["gameDate"]).sort_values("gameDate").reset_index(drop=True)
    df = df[df["gameDate"] >= START_DATE].copy()

    required = [TARGET, *FEATURE_MAP.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"MasterDataset is missing required columns: {missing}")

    model_df = pd.DataFrame(
        {
            "gameDate": df["gameDate"],
            **{feature: pd.to_numeric(df[column], errors="coerce") for feature, column in FEATURE_MAP.items()},
            HOME_FEATURE: 1.0,
            TARGET: pd.to_numeric(df[TARGET], errors="coerce"),
        }
    )
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    return model_df.dropna(subset=[TARGET]).reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy(), split_idx


def prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Use only training-set medians and scales so the test window cannot leak backward.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)

    means = X_train[SCALED_FEATURES].mean()
    scales = X_train[SCALED_FEATURES].std(ddof=0).replace(0.0, 1.0)
    X_train.loc[:, SCALED_FEATURES] = (X_train[SCALED_FEATURES] - means) / scales
    X_test.loc[:, SCALED_FEATURES] = (X_test[SCALED_FEATURES] - means) / scales
    return X_train, X_test, means, scales


def original_coefficients(model: Lasso, means: pd.Series, scales: pd.Series) -> pd.Series:
    # Convert coefficients back from the scaled feature space so the printed values are interpretable.
    scaled = pd.Series(model.coef_, index=FEATURES, dtype=float)
    coefficients = scaled.copy()
    coefficients.loc[SCALED_FEATURES] = scaled.loc[SCALED_FEATURES] / scales
    coefficients[HOME_FEATURE] -= float((scaled.loc[SCALED_FEATURES] * means / scales).sum())
    return coefficients.sort_values(key=np.abs, ascending=False)


def main() -> None:
    model_df = load_model_frame()
    train_df, test_df, split_idx = split_data(model_df)

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()
    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    X_train, X_test, means, scales = prepare_features(X_train, X_test)
    model = Lasso(alpha=LASSO_ALPHA, fit_intercept=False, max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    coefficients = original_coefficients(model, means, scales)
    non_zero = coefficients[coefficients.abs() > 1e-8]

    print(f"Rows: {len(model_df)}  |  {model_df['gameDate'].min()} -> {model_df['gameDate'].max()}")
    print(f"Start date filter: {START_DATE.date()}")
    print(
        f"Chronological split: {len(train_df)} train / {len(test_df)} test "
        f"({len(train_df) / len(model_df):.1%} / {len(test_df) / len(model_df):.1%})"
    )
    print(f"Split index: {split_idx}")
    print(f"Train end: {train_df['gameDate'].iloc[-1]}")
    print(f"Test start: {test_df['gameDate'].iloc[0]}")
    print(f"Lasso alpha: {LASSO_ALPHA}")
    print()
    print("Feature set:")
    for feature_name, source_column in FEATURE_MAP.items():
        print(f"  {feature_name:<18} <- {source_column}")
    print("  home_indicator      <- constant 1.0 baseline")
    print()
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Non-zero coefficients: {len(non_zero)} / {len(coefficients)}")
    print()
    print("Coefficients (original units):")
    print(coefficients.round(4).to_string())


if __name__ == "__main__":
    main()
