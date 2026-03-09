import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. LOAD & FILTER
# ─────────────────────────────────────────────
from pathlib import Path

data_path = Path(__file__).resolve().parent / "Data" / "final" / "master_games_model_ready_regular_season.csv"
df = pd.read_csv(
    data_path,
    low_memory=False
)
df["gameDate"] = pd.to_datetime(df["gameDate"])
df = df.sort_values("gameDate").reset_index(drop=True)
df = df[df["gameDate"] >= "2020-07-30"].copy()
df = df[df["Total"] > 0].copy()
print(f"Rows: {len(df)}  |  {df['gameDate'].min().date()} → {df['gameDate'].max().date()}")

# ─────────────────────────────────────────────
# 2. BUILD ALL PER-TEAM ROLLING FEATURES
# ─────────────────────────────────────────────
all_rows = []
for _, row in df.iterrows():
    all_rows.append({
        "gameDate": row["gameDate"], "teamId": row["HomeTeamId"],
        "pts": row["HomePTS"], "pts_against": row["AwayPTS"],
        "total": row["Total"], "poss": row["Home_poss"],
        "eFG": row["Home_eFG"], "TS": row["Home_TS"],
        "TOV_rate": row["Home_TOV_rate"], "FT_rate": row["Home_FT_rate"],
        "threePA_rate": row["Home_3PA_rate"],
    })
    all_rows.append({
        "gameDate": row["gameDate"], "teamId": row["AwayTeamId"],
        "pts": row["AwayPTS"], "pts_against": row["HomePTS"],
        "total": row["Total"], "poss": row["Away_poss"],
        "eFG": row["Away_eFG"], "TS": row["Away_TS"],
        "TOV_rate": row["Away_TOV_rate"], "FT_rate": row["Away_FT_rate"],
        "threePA_rate": row["Away_3PA_rate"],
    })

tdf = pd.DataFrame(all_rows).sort_values(["teamId", "gameDate"]).reset_index(drop=True)

stat_cols = ["pts", "pts_against", "total", "poss", "eFG", "TS", "TOV_rate", "FT_rate", "threePA_rate"]
windows   = [5, 10, 20]

for col in stat_cols:
    for w in windows:
        tdf[f"roll{w}_{col}"] = tdf.groupby("teamId")[col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).mean())
    for span in [10, 20]:
        tdf[f"ewm{span}_{col}"] = tdf.groupby("teamId")[col].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean())

roll_cols = [c for c in tdf.columns if c.startswith("roll") or c.startswith("ewm")]

home = (tdf.rename(columns={"teamId": "HomeTeamId"})
           .rename(columns={c: f"home_{c}" for c in roll_cols}))
away = (tdf.rename(columns={"teamId": "AwayTeamId"})
           .rename(columns={c: f"away_{c}" for c in roll_cols}))

h_cols = [f"home_{c}" for c in roll_cols]
a_cols = [f"away_{c}" for c in roll_cols]

df = df.merge(home[["gameDate","HomeTeamId"]+h_cols], on=["gameDate","HomeTeamId"], how="left")
df = df.merge(away[["gameDate","AwayTeamId"]+a_cols], on=["gameDate","AwayTeamId"], how="left")

# Combined (home + away) features
for col in stat_cols:
    for w in windows:
        df[f"combined_roll{w}_{col}"] = df[f"home_roll{w}_{col}"] + df[f"away_roll{w}_{col}"]
    for span in [10, 20]:
        df[f"combined_ewm{span}_{col}"] = df[f"home_ewm{span}_{col}"] + df[f"away_ewm{span}_{col}"]

# ─────────────────────────────────────────────
# 3. OTHER ENGINEERED FEATURES (from v1/v2)
# ─────────────────────────────────────────────
df["avg_last5_pace"]     = (df["Home_last5_pts_for"]+df["Home_last5_pts_against"]+
                             df["Away_last5_pts_for"]+df["Away_last5_pts_against"])
df["avg_last10_pace"]    = (df["Home_last10_pts_for"]+df["Home_last10_pts_against"]+
                             df["Away_last10_pts_for"]+df["Away_last10_pts_against"])
df["opp_adj_total"]      = (df["Home_last10_pts_for"]+df["Away_last10_pts_against"]+
                             df["Away_last10_pts_for"]+df["Home_last10_pts_against"])
df["either_b2b"]         = ((df["Home_b2b"]==1)|(df["Away_b2b"]==1)).astype(int)
df["both_b2b"]           = ((df["Home_b2b"]==1)&(df["Away_b2b"]==1)).astype(int)
df["home_winpct"]        = df["Home_seasonWins"]/(df["Home_seasonWins"]+df["Home_seasonLosses"]+1e-9)
df["away_winpct"]        = df["Away_seasonWins"]/(df["Away_seasonWins"]+df["Away_seasonLosses"]+1e-9)
df["log_Total"]          = np.log(df["Total"])

# H2H rolling total
df["pair"] = df.apply(lambda r: tuple(sorted([r["HomeTeamId"], r["AwayTeamId"]])), axis=1)
df["h2h_avg_total"] = df.groupby("pair")["Total"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df["h2h_avg_total"] = df["h2h_avg_total"].fillna(df["Total"].mean())

# ─────────────────────────────────────────────
# 4. FEATURE SETS
# ─────────────────────────────────────────────
# Core dataset rolling features (pre-built, most reliable)
FEATURES_DATASET = [
    "Home_last5_pts_for","Home_last5_pts_against",
    "Away_last5_pts_for","Away_last5_pts_against",
    "Home_last10_pts_for","Home_last10_pts_against",
    "Away_last10_pts_for","Away_last10_pts_against",
    "avg_last5_pace","avg_last10_pace","opp_adj_total",
    "Home_b2b","Away_b2b","either_b2b","both_b2b","rest_diff",
    "home_winpct","away_winpct",
    "Home_last5_margin","Away_last5_margin",
    "diff_last5_pts_for","diff_last5_pts_against",
    "diff_last10_pts_for","diff_last10_pts_against",
    "h2h_avg_total",
]

# New per-team rolling signals
FEATURES_PERTEAM = [
    "combined_roll5_total","combined_roll10_total","combined_roll20_total",
    "combined_ewm10_total","combined_ewm20_total",
    "combined_roll5_pts_against","combined_roll10_pts_against","combined_roll20_pts_against",
    "combined_ewm10_pts_against",
    "combined_roll10_poss","combined_roll20_poss","combined_ewm10_poss","combined_ewm20_poss",
    "combined_roll10_pts","combined_roll20_pts","combined_ewm10_pts",
]

# Efficiency features (lower correlation but potentially useful)
FEATURES_EFFICIENCY = [
    "combined_roll10_eFG","combined_roll20_eFG","combined_ewm10_eFG",
    "combined_roll10_TS","combined_roll20_TS",
    "combined_roll10_TOV_rate","combined_roll10_FT_rate","combined_roll10_threePA_rate",
]

FEATURES_V3_CORE = FEATURES_DATASET + FEATURES_PERTEAM
FEATURES_V3_FULL = FEATURES_V3_CORE + FEATURES_EFFICIENCY

print(f"\nFeature set sizes:")
print(f"  Dataset rolling (v1-style): {len(FEATURES_DATASET)}")
print(f"  + Per-team rolling (v3):    {len(FEATURES_V3_CORE)}")
print(f"  + Efficiency (v3 full):     {len(FEATURES_V3_FULL)}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
SPLIT_DATE = "2026-01-14"
train = df[df["gameDate"] < SPLIT_DATE].copy()
test  = df[df["gameDate"] >= SPLIT_DATE].copy()
print(f"\nTrain: {len(train)}  |  Test: {len(test)}")

def fill_na(X, ref):
    """Fill nulls with training column medians."""
    med = np.nanmedian(ref, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X

# ─────────────────────────────────────────────
# 6. HELPERS
# ─────────────────────────────────────────────
def tscv_mae(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[tr_idx].copy(), X[val_idx].copy()
        X_tr  = fill_na(X_tr,  X_tr)
        X_val = fill_na(X_val, X_tr)
        model.fit(X_tr, y[tr_idx])
        maes.append(mean_absolute_error(y[val_idx], model.predict(X_val)))
    return np.mean(maes), np.std(maes)

def get_Xy(feature_cols, df_train, df_test, log_target=False):
    X_tr = df_train[feature_cols].values.copy().astype(float)
    X_te = df_test[feature_cols].values.copy().astype(float)
    X_tr = fill_na(X_tr, X_tr)
    X_te = fill_na(X_te, X_tr)
    y_tr = df_train["log_Total"].values if log_target else df_train["Total"].values
    y_te = df_test["Total"].values
    return X_tr, X_te, y_tr, y_te

def eval_model(name, model, feature_cols, df_train, df_test, log_target=False):
    X_tr, X_te, y_tr, y_te = get_Xy(feature_cols, df_train, df_test, log_target)
    cv_mae, cv_std = tscv_mae(model, X_tr, y_tr)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    if log_target:
        preds = np.exp(preds)
    test_mae = mean_absolute_error(y_te, preds)
    tag = " [log]" if log_target else ""
    print(f"  {(name+tag):<55}  CV: {cv_mae:.2f}±{cv_std:.2f}  Test: {test_mae:.2f}")
    return {"model": name+tag, "cv_mae": cv_mae, "cv_std": cv_std,
            "test_mae": test_mae, "preds": preds}

# ─────────────────────────────────────────────
# 7. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "="*72)
print("MODEL COMPARISON")
print("="*72)
results = []

# Naive
naive_val = train["Total"].mean()
naive_mae = mean_absolute_error(test["Total"], np.full(len(test), naive_val))
print(f"  {'Naive (season mean)':<57}               Test: {naive_mae:.2f}")
results.append({"model":"Naive","cv_mae":np.nan,"cv_std":np.nan,
                "test_mae":naive_mae,"preds":np.full(len(test),naive_val)})

print("\n── SIMPLE / LINEAR ──")
r = eval_model("OLS (dataset features)",
               LinearRegression(), FEATURES_DATASET, train, test)
results.append(r)

r = eval_model("Ridge (dataset features)",
               Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=10))]),
               FEATURES_DATASET, train, test)
results.append(r)

r = eval_model("OLS (v3 core)",
               LinearRegression(), FEATURES_V3_CORE, train, test)
results.append(r)

r = eval_model("Ridge (v3 core)",
               Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=100))]),
               FEATURES_V3_CORE, train, test)
results.append(r)

r = eval_model("Ridge (v3 full)",
               Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=100))]),
               FEATURES_V3_FULL, train, test)
results.append(r)

r = eval_model("OLS log(Total) (v3 core)",
               LinearRegression(), FEATURES_V3_CORE, train, test, log_target=True)
results.append(r)

r = eval_model("ElasticNet (v3 core)",
               Pipeline([("sc",StandardScaler()),
                          ("m",ElasticNet(alpha=0.3,l1_ratio=0.5,max_iter=5000))]),
               FEATURES_V3_CORE, train, test)
results.append(r)

print("\n── POLYNOMIAL TERMS (on top features) ──")
TOP_FEATS = ["combined_roll20_total","combined_ewm10_total","avg_last10_pace",
             "opp_adj_total","h2h_avg_total","combined_ewm10_poss","home_winpct","away_winpct"]

r = eval_model("OLS + poly2 interactions (top feats)",
               Pipeline([("sc",StandardScaler()),
                          ("poly",PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)),
                          ("m",Ridge(alpha=100))]),
               TOP_FEATS, train, test)
results.append(r)

print("\n── TREE / ENSEMBLE ──")
r = eval_model("Random Forest (v3 core)",
               RandomForestRegressor(n_estimators=300, max_depth=6,
                                     min_samples_leaf=10, random_state=42, n_jobs=-1),
               FEATURES_V3_CORE, train, test)
results.append(r)

r = eval_model("GradBoost (v3 core, tuned)",
               GradientBoostingRegressor(n_estimators=300, learning_rate=0.04,
                                          max_depth=3, min_samples_leaf=10,
                                          subsample=0.8, random_state=42),
               FEATURES_V3_CORE, train, test)
results.append(r)

print("\n── LASSO FEATURE SELECTION ON V3 ──")
X_tr_v3, X_te_v3, y_tr_raw, y_te = get_Xy(FEATURES_V3_CORE, train, test)
sc_l = StandardScaler()
X_l  = sc_l.fit_transform(X_tr_v3)
lasso = Lasso(alpha=0.3, max_iter=10000).fit(X_l, y_tr_raw)
coef  = pd.Series(lasso.coef_, index=FEATURES_V3_CORE)
FEATURES_LASSO_V3 = coef[coef != 0].sort_values(key=abs, ascending=False).index.tolist()
print(f"  Lasso kept {len(FEATURES_LASSO_V3)}/{len(FEATURES_V3_CORE)} features: {FEATURES_LASSO_V3}")

r = eval_model("OLS (lasso-selected v3)",
               LinearRegression(), FEATURES_LASSO_V3, train, test)
results.append(r)
r = eval_model("Ridge (lasso-selected v3, alpha=100)",
               Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=100))]),
               FEATURES_LASSO_V3, train, test)
results.append(r)

print("\n── STACKING (meta-learner combining best models) ──")
X_tr_stack, X_te_stack, y_tr_raw, y_te = get_Xy(FEATURES_V3_CORE, train, test)

estimators = [
    ("ridge",  Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=100))])),
    ("ols",    LinearRegression()),
    ("rf",     RandomForestRegressor(n_estimators=200, max_depth=5,
                                      min_samples_leaf=15, random_state=42, n_jobs=-1)),
    ("gb",     GradientBoostingRegressor(n_estimators=200, learning_rate=0.04,
                                          max_depth=2, min_samples_leaf=15,
                                          subsample=0.8, random_state=42)),
]
stacker = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=10),
    cv=5,
    passthrough=False,
)
r = eval_model("Stacking (Ridge+OLS+RF+GB → Ridge meta)",
               stacker, FEATURES_V3_CORE, train, test)
results.append(r)

# ─────────────────────────────────────────────
# 8. RIDGE ALPHA SEARCH ON BEST FEATURE SET
# ─────────────────────────────────────────────
print("\n── RIDGE ALPHA SEARCH (v3 core) ──")
best_alpha, best_cv_val = None, np.inf
for alpha in [0.1, 1, 10, 50, 100, 200, 500, 1000]:
    m = Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=alpha))])
    cv_v, cv_s = tscv_mae(m, X_tr_v3, y_tr_raw)
    print(f"    alpha={alpha:<6}  CV: {cv_v:.3f}±{cv_s:.3f}")
    if cv_v < best_cv_val:
        best_cv_val, best_alpha = cv_v, alpha

print(f"  → Best alpha: {best_alpha}")
r = eval_model(f"Ridge TUNED alpha={best_alpha} (v3 core)",
               Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=best_alpha))]),
               FEATURES_V3_CORE, train, test)
results.append(r)

# ─────────────────────────────────────────────
# 9. RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*72)
print("FINAL RESULTS SUMMARY (sorted by Test MAE)")
print("="*72)
res_df = pd.DataFrame(results)[["model","cv_mae","test_mae"]].sort_values("test_mae")
print(res_df.to_string(index=False))

# ─────────────────────────────────────────────
# 10. BEST MODEL INTERPRETATION
# ─────────────────────────────────────────────
honest = res_df.copy()
best_name = honest.iloc[0]["model"]
print(f"\nBest model: {best_name}  →  Test MAE: {honest.iloc[0]['test_mae']:.3f} pts")

# Ridge coefficients for best linear model
best_ridge_name = [r for r in res_df["model"] if "Ridge" in r and "Stacking" not in r][0]
best_ridge_feats = FEATURES_LASSO_V3 if "lasso" in best_ridge_name else \
                   FEATURES_V3_CORE  if "v3 core" in best_ridge_name else FEATURES_DATASET
sc_f = StandardScaler()
X_f  = sc_f.fit_transform(train[best_ridge_feats].fillna(train[best_ridge_feats].median()).values)
coef_final = Ridge(alpha=best_alpha).fit(X_f, train["Total"].values)
coef_series = pd.Series(coef_final.coef_, index=best_ridge_feats).sort_values(key=abs, ascending=False)
print(f"\nTop 10 Ridge coefficients ({best_ridge_name}):")
print(coef_series.head(10).round(3).to_string())

# ─────────────────────────────────────────────
# 11. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# (a) Model comparison
plot_df = res_df.dropna(subset=["test_mae"])
colors = ["green" if i == 0 else "steelblue" for i in range(len(plot_df))]
axes[0].barh(plot_df["model"], plot_df["test_mae"], color=colors)
axes[0].axvline(naive_mae, color="red", linestyle="--", lw=1.5, label=f"Naive={naive_mae:.1f}")
axes[0].set_xlabel("Test MAE (points)")
axes[0].set_title("All Models – Test MAE\n(green = best)")
axes[0].legend(fontsize=8)
axes[0].invert_yaxis()

# (b) Ridge coefficients
top_coef = coef_series.head(12)
bar_colors = ["tomato" if v < 0 else "steelblue" for v in top_coef.values]
axes[1].barh(top_coef.index[::-1], top_coef.values[::-1], color=bar_colors[::-1])
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("Coefficient (standardized features)")
axes[1].set_title(f"Top Ridge Coefficients\n(blue=+, red=−)")

# (c) Predicted vs Actual
best_preds = next(r["preds"] for r in results if r["model"] == best_name)
axes[2].scatter(test["Total"], best_preds, alpha=0.5, s=18, color="green")
mn = min(test["Total"].min(), min(best_preds)) - 5
mx = max(test["Total"].max(), max(best_preds)) + 5
axes[2].plot([mn,mx],[mn,mx],"r--",lw=1)
axes[2].set_xlabel("Actual Total")
axes[2].set_ylabel("Predicted Total")
axes[2].set_title(f"Predicted vs Actual\n{best_name}\nMAE={honest.iloc[0]['test_mae']:.2f} pts")

plt.tight_layout()
output_path = Path(__file__).resolve().parent / "total_model_v3_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")

# ─────────────────────────────────────────────
# 12. PAPER SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*72)
print("PAPER SUMMARY")
print("="*72)
print(f"  Best model      : {best_name}")
print(f"  Test MAE        : {honest.iloc[0]['test_mae']:.3f} pts")
print(f"  Naive baseline  : {naive_mae:.3f} pts")
print(f"  Improvement     : {naive_mae - honest.iloc[0]['test_mae']:.3f} pts  "
      f"({(naive_mae - honest.iloc[0]['test_mae'])/naive_mae*100:.1f}%)")
print(f"\n  Key new features (top correlations with Total):")
for feat in ["combined_roll20_total","combined_ewm10_total","combined_roll10_pts_against","combined_ewm10_poss"]:
    print(f"    {feat}: r={df[feat].corr(df['Total']):.3f}")