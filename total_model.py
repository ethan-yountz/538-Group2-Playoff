"""
total_model_v3.py
=================
Predicting NBA Game Total Points (Home Points + Away Points)
Updated for MasterDataset.csv (4+ seasons, 2021–2026)

  - Home_seasonWins / Home_seasonLosses are null for seasons 2021-2023 in the
    new dataset, so win percentage is computed from scratch for all seasons
  - is_playoff column available; playoff games are excluded from training
  - More training data (5,843 games vs 1,988) improves CV MAE from ~15.03 -> ~14.78
  - Best alpha updated to 500 (more data = more regularization still optimal)
  - poss_avg leakage check re-confirmed on new dataset (r=0.482 -> 0.004 when lagged)

Model: Ridge Regression (alpha=500) on 13 Lasso-selected features
Best Test MAE: 14.694 (vs Naive baseline 15.795)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
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
df = pd.read_csv("MasterDataset.csv", low_memory=False)
df["gameDate"] = pd.to_datetime(df["gameDate"])
df = df.sort_values("gameDate").reset_index(drop=True)

# Drop rows with no score (data errors)
df = df[df["Total"] > 0].copy()

# Exclude playoff games — different pace/intensity distorts regular-season model
# is_playoff is NaN for older seasons (treated as regular season)
df = df[df["is_playoff"] != True].copy()
df = df.reset_index(drop=True)

print(f"Rows after filtering: {len(df)}")
print(f"Date range: {df['gameDate'].min().date()} -> {df['gameDate'].max().date()}")
print(f"Total: mean={df['Total'].mean():.1f}  std={df['Total'].std():.1f}  skew={df['Total'].skew():.3f}")

# ─────────────────────────────────────────────
# 2. LEAKAGE CHECK: poss_avg
# ─────────────────────────────────────────────
# poss_avg uses current-game possessions — post-game information.
# Confirmed: r=0.482 with Total, but lagged r≈0.004.
# EXCLUDED from all models.
poss_corr        = df["poss_avg"].corr(df["Total"])
poss_corr_lagged = df["poss_avg"].shift(1).corr(df["Total"])
print(f"\nposs_avg correlation with Total:        {poss_corr:.3f}  (current game — LEAKAGE)")
print(f"poss_avg lagged 1 game correlation:    {poss_corr_lagged:.3f}  (no signal — confirms leakage)")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# ── 3a. Season win percentage (computed from scratch) ────────────────────────
# Home_seasonWins / Home_seasonLosses are null for seasons 2021-2023 in the
# dataset, so we recompute win percentage for every team-season from the game log.
# Season = calendar year of Oct-Dec start (e.g. Oct 2021 -> season 2021)
df["season"] = df["gameDate"].dt.year.where(
    df["gameDate"].dt.month >= 9, df["gameDate"].dt.year - 1
)

game_results = []
for _, r in df.iterrows():
    game_results.append({
        "gameDate": r["gameDate"], "season": r["season"],
        "team": r["HomeTeam"], "win": int(r["HomePTS"] > r["AwayPTS"])
    })
    game_results.append({
        "gameDate": r["gameDate"], "season": r["season"],
        "team": r["AwayTeam"],  "win": int(r["AwayPTS"] > r["HomePTS"])
    })

gdf = (pd.DataFrame(game_results)
         .sort_values(["team", "season", "gameDate"])
         .reset_index(drop=True))

# Cumulative wins and games BEFORE each game (shift(1) ensures no current-game leakage)
gdf["cum_wins"]  = gdf.groupby(["team","season"])["win"].transform(
    lambda x: x.shift(1).expanding().sum())
gdf["cum_games"] = gdf.groupby(["team","season"])["win"].transform(
    lambda x: x.shift(1).expanding().count())
gdf["winpct"] = (gdf["cum_wins"] / gdf["cum_games"].replace(0, np.nan)).fillna(0.5)

# Merge back: first game of each season fills 0.5 (no prior history)
df = df.merge(
    gdf.rename(columns={"team":"HomeTeam","winpct":"home_winpct"})
       [["gameDate","HomeTeam","home_winpct"]],
    on=["gameDate","HomeTeam"], how="left"
)
df = df.merge(
    gdf.rename(columns={"team":"AwayTeam","winpct":"away_winpct"})
       [["gameDate","AwayTeam","away_winpct"]],
    on=["gameDate","AwayTeam"], how="left"
)

# ── 3b. Per-team game log: rolling and EWM pace features ─────────────────────
# Stack every game twice (once per team) to compute per-team rolling stats
# across both home and away appearances.
team_log = []
for _, row in df.iterrows():
    team_log.append({
        "gameDate": row["gameDate"], "team": row["HomeTeam"],
        "total": row["Total"],       "poss":  row["Home_poss"]
    })
    team_log.append({
        "gameDate": row["gameDate"], "team": row["AwayTeam"],
        "total": row["Total"],       "poss":  row["Away_poss"]
    })

tdf = (pd.DataFrame(team_log)
         .sort_values(["team","gameDate"])
         .reset_index(drop=True))

# Rolling and EWM stats — shift(1) ensures current game is excluded (no leakage)
for col in ["total", "poss"]:
    for w in [10, 20]:
        tdf[f"roll{w}_{col}"] = tdf.groupby("team")[col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        tdf[f"ewm{w}_{col}"]  = tdf.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=w, min_periods=1).mean())

TEAM_STAT_COLS = [c for c in tdf.columns if c not in ["gameDate","team","total","poss"]]

# Merge per-team stats back onto game-level data
df = df.merge(
    tdf.rename(columns={"team":"HomeTeam"})
       [["gameDate","HomeTeam"] + TEAM_STAT_COLS]
       .rename(columns={c: f"hpt_{c}" for c in TEAM_STAT_COLS}),
    on=["gameDate","HomeTeam"], how="left"
)
df = df.merge(
    tdf.rename(columns={"team":"AwayTeam"})
       [["gameDate","AwayTeam"] + TEAM_STAT_COLS]
       .rename(columns={c: f"apt_{c}" for c in TEAM_STAT_COLS}),
    on=["gameDate","AwayTeam"], how="left"
)

# Combined (home + away) features
for c in TEAM_STAT_COLS:
    df[f"comb_{c}"] = df[f"hpt_{c}"] + df[f"apt_{c}"]

# ── 3c. Features derived from existing dataset rolling columns ────────────────
df["avg_last5_pace"]  = (df["Home_last5_pts_for"]  + df["Home_last5_pts_against"] +
                          df["Away_last5_pts_for"]  + df["Away_last5_pts_against"])
df["avg_last10_pace"] = (df["Home_last10_pts_for"] + df["Home_last10_pts_against"] +
                          df["Away_last10_pts_for"] + df["Away_last10_pts_against"])
df["opp_adj_total"]   = (df["Home_last10_pts_for"] + df["Away_last10_pts_against"] +
                          df["Away_last10_pts_for"] + df["Home_last10_pts_against"])
df["both_b2b"]        = ((df["Home_b2b"] == 1) & (df["Away_b2b"] == 1)).astype(int)
df["either_b2b"]      = ((df["Home_b2b"] == 1) | (df["Away_b2b"] == 1)).astype(int)

# ── 3d. Head-to-head rolling Total ───────────────────────────────────────────
df["pair"] = df.apply(
    lambda r: tuple(sorted([r["HomeTeamId"], r["AwayTeamId"]])), axis=1)
df["h2h_avg_total"] = df.groupby("pair")["Total"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df["h2h_avg_total"] = df["h2h_avg_total"].fillna(df["Total"].mean())

# ── 3e. Log-transformed target ───────────────────────────────────────────────
df["log_Total"] = np.log(df["Total"])

# ─────────────────────────────────────────────
# 4. FEATURE SETS
# ─────────────────────────────────────────────

# 13 Lasso-selected features (same as v2, re-validated on new dataset)
FEATURES_LASSO = [
    "comb_roll20_total",       # per-team rolling 20-game avg of game Totals (r=0.359)
    "comb_ewm20_total",        # per-team EWM avg of game Totals, span=20 (r=0.373)
    "comb_ewm20_poss",         # per-team EWM possessions/game, span=20 (r=0.275)
    "avg_last10_pace",         # combined scoring pace last 10 games
    "Away_last5_pts_for",      # away team recent offense
    "comb_ewm10_poss",         # per-team EWM possessions/game, span=10 (r=0.256)
    "h2h_avg_total",           # head-to-head historical Total
    "Home_last10_pts_against", # home team recent defense
    "Home_b2b",                # home team on back-to-back (fatigue)
    "home_winpct",             # home team season win % (computed from scratch)
    "both_b2b",                # both teams on back-to-back
    "Home_last5_pts_against",  # home team recent defense (short window)
    "opp_adj_total",           # opponent-adjusted combined scoring
]

# Extended feature set (all engineered features)
FEATURES_V3 = FEATURES_LASSO + [
    "comb_roll10_total", "comb_ewm10_total",
    "comb_roll10_poss",  "comb_roll20_poss",
    "avg_last5_pace",    "away_winpct",
    "either_b2b",        "rest_diff",
    "Home_last5_pts_for","Away_last10_pts_for",
    "Away_last10_pts_against",
]

print(f"\nLasso feature set size: {len(FEATURES_LASSO)}")
print(f"Extended feature set size: {len(FEATURES_V3)}")

# Top correlations
print("\nTop feature correlations with Total:")
for feat in ["comb_ewm20_total","comb_roll20_total","comb_ewm10_total",
             "comb_ewm20_poss","avg_last10_pace","h2h_avg_total"]:
    print(f"  {feat}: r={df[feat].corr(df['Total']):.3f}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
SPLIT_DATE = "2026-01-14"
train = df[df["gameDate"] < SPLIT_DATE].copy()
test  = df[df["gameDate"] >= SPLIT_DATE].copy()
print(f"\nTrain: {len(train.dropna(subset=FEATURES_LASSO))}  |  Test: {len(test.dropna(subset=FEATURES_LASSO))}")

# ─────────────────────────────────────────────
# 6. HELPERS
# ─────────────────────────────────────────────
def prep(feature_cols, df_train, df_test, log_target=False):
    tr = df_train.dropna(subset=feature_cols + ["Total"]).copy()
    te = df_test.dropna(subset=feature_cols + ["Total"]).copy()
    X_tr = tr[feature_cols].values.astype(float)
    X_te = te[feature_cols].values.astype(float)
    # Fill any residual NaNs with training medians
    med = np.nanmedian(X_tr, axis=0)
    for arr in [X_tr, X_te]:
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(med, idx[1])
    y_tr = tr["log_Total"].values if log_target else tr["Total"].values
    y_te = te["Total"].values
    return X_tr, X_te, y_tr, y_te, tr, te

def tscv_mae(model, X, y, n=5):
    tscv = TimeSeriesSplit(n_splits=n)
    return np.mean([
        mean_absolute_error(y[v], model.fit(X[t], y[t]).predict(X[v]))
        for t, v in tscv.split(X)
    ])

def evaluate(name, model, feature_cols, df_train, df_test, log_target=False):
    X_tr, X_te, y_tr, y_te, _, _ = prep(feature_cols, df_train, df_test, log_target)
    cv  = tscv_mae(model, X_tr, y_tr)
    model.fit(X_tr, y_tr)
    preds = np.exp(model.predict(X_te)) if log_target else model.predict(X_te)
    test_mae = mean_absolute_error(y_te, preds)
    tag = " [log]" if log_target else ""
    print(f"  {(name+tag):<55}  CV: {cv:.2f}  Test: {test_mae:.2f}")
    return {"model": name+tag, "cv_mae": cv, "test_mae": test_mae, "preds": preds}

# ─────────────────────────────────────────────
# 7. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
results = []

# Naive baseline
naive_pred = train["Total"].mean()
naive_mae  = mean_absolute_error(test["Total"].dropna(), np.full(len(test["Total"].dropna()), naive_pred))
print(f"  {'Naive (season mean)':<57}               Test: {naive_mae:.2f}")
results.append({"model":"Naive","cv_mae":np.nan,"test_mae":naive_mae,
                "preds":np.full(len(test),naive_pred)})

print("\n── LINEAR MODELS ──")

r = evaluate("OLS (lasso features)",
             LinearRegression(), FEATURES_LASSO, train, test)
results.append(r)

r = evaluate("OLS log(Total) (lasso features)",
             LinearRegression(), FEATURES_LASSO, train, test, log_target=True)
results.append(r)

for alpha in [50, 100, 200, 500, 1000]:
    r = evaluate(f"Ridge alpha={alpha} (lasso features)",
                 Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=alpha))]),
                 FEATURES_LASSO, train, test)
    results.append(r)

r = evaluate("ElasticNet (a=0.3, l1=0.5) (lasso features)",
             Pipeline([("sc",StandardScaler()),
                        ("m",ElasticNet(alpha=0.3,l1_ratio=0.5,max_iter=5000))]),
             FEATURES_LASSO, train, test)
results.append(r)

r = evaluate("Ridge alpha=500 (v3 extended features)",
             Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=500))]),
             FEATURES_V3, train, test)
results.append(r)

print("\n── POLYNOMIAL INTERACTIONS ──")
TOP8 = ["comb_ewm20_total","comb_roll20_total","avg_last10_pace",
        "opp_adj_total","h2h_avg_total","comb_ewm20_poss",
        "home_winpct","Home_b2b"]
r = evaluate("Poly2 interactions + Ridge alpha=100 (top 8)",
             Pipeline([("sc",StandardScaler()),
                        ("poly",PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)),
                        ("m",Ridge(alpha=100))]),
             TOP8, train, test)
results.append(r)

print("\n── TREE / ENSEMBLE ──")
r = evaluate("Random Forest (300 trees, d=6, leaf=10)",
             RandomForestRegressor(n_estimators=300,max_depth=6,
                                    min_samples_leaf=10,random_state=42,n_jobs=-1),
             FEATURES_LASSO, train, test)
results.append(r)

r = evaluate("GradBoost (n=200, lr=0.04, d=2, leaf=15)",
             GradientBoostingRegressor(n_estimators=200,learning_rate=0.04,
                                        max_depth=2,min_samples_leaf=15,
                                        subsample=0.8,random_state=42),
             FEATURES_LASSO, train, test)
results.append(r)

print("\n── NEURAL NETWORK ──")
r = evaluate("MLP (64->32, alpha=0.01, early_stop)",
             Pipeline([("sc",StandardScaler()),
                        ("m",MLPRegressor(hidden_layer_sizes=(64,32),alpha=0.01,
                                           early_stopping=True,max_iter=500,random_state=42))]),
             FEATURES_LASSO, train, test)
results.append(r)

print("\n── STACKING ──")
base_learners = [
    ("ridge", Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=500))])),
    ("ols",   LinearRegression()),
    ("rf",    RandomForestRegressor(n_estimators=200,max_depth=5,
                                     min_samples_leaf=15,random_state=42,n_jobs=-1)),
    ("gb",    GradientBoostingRegressor(n_estimators=200,learning_rate=0.04,
                                         max_depth=2,min_samples_leaf=15,
                                         subsample=0.8,random_state=42)),
]
r = evaluate("Stacking (Ridge+OLS+RF+GB -> Ridge meta)",
             StackingRegressor(estimators=base_learners,
                                final_estimator=Ridge(alpha=10), cv=5),
             FEATURES_LASSO, train, test)
results.append(r)

# ─────────────────────────────────────────────
# 8. LASSO FEATURE SELECTION (re-run on new dataset)
# ─────────────────────────────────────────────
print("\n── LASSO FEATURE SELECTION ──")
X_tr_v3, X_te_v3, y_tr_raw, y_te_v3, tr_v3, te_v3 = prep(FEATURES_V3, train, test)
sc_l  = StandardScaler()
X_l   = sc_l.fit_transform(X_tr_v3)
lasso = Lasso(alpha=0.3, max_iter=10000).fit(X_l, y_tr_raw)
coef  = pd.Series(lasso.coef_, index=FEATURES_V3)
FEATURES_LASSO_NEW = coef[coef != 0].sort_values(key=abs, ascending=False).index.tolist()
print(f"  Lasso retained {len(FEATURES_LASSO_NEW)}/{len(FEATURES_V3)} features:")
print(f"  {FEATURES_LASSO_NEW}")

r = evaluate("Ridge alpha=500 (new lasso-selected features)",
             Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=500))]),
             FEATURES_LASSO_NEW, train, test)
results.append(r)

# ─────────────────────────────────────────────
# 9. RESULTS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL RESULTS (sorted by Test MAE)")
print("="*70)
res_df = pd.DataFrame(results)[["model","cv_mae","test_mae"]].sort_values("test_mae")
print(res_df.to_string(index=False))

best      = res_df.iloc[0]
best_name = best["model"]
print(f"\nBest model: {best_name}")
print(f"Test MAE:   {best['test_mae']:.3f} pts")
print(f"Naive MAE:  {naive_mae:.3f} pts")
print(f"Improvement:{naive_mae - best['test_mae']:.3f} pts ({(naive_mae-best['test_mae'])/naive_mae*100:.1f}%)")

# ─────────────────────────────────────────────
# 10. BEST MODEL COEFFICIENTS
# ─────────────────────────────────────────────
print("\n── BEST MODEL: Ridge alpha=500, Lasso-selected features ──")
X_tr_f, X_te_f, y_tr_f, y_te_f, _, _ = prep(FEATURES_LASSO, train, test)
sc_f = StandardScaler()
X_f  = sc_f.fit_transform(X_tr_f)
best_model = Ridge(alpha=500).fit(X_f, y_tr_f)
coef_df = pd.Series(best_model.coef_, index=FEATURES_LASSO).sort_values(key=abs, ascending=False)
print(f"  Intercept: {best_model.intercept_:.2f}")
print("\n  Standardized coefficients:")
for feat, val in coef_df.items():
    print(f"    {feat:<35} {val:+.3f}")

# ─────────────────────────────────────────────
# 11. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# (a) Model comparison bar chart
plot_df = res_df.dropna(subset=["test_mae"])
colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(plot_df))]
axes[0].barh(plot_df["model"], plot_df["test_mae"], color=colors)
axes[0].axvline(naive_mae, color="red", linestyle="--", lw=1.5,
                label=f"Naive = {naive_mae:.1f}")
axes[0].set_xlabel("Test MAE (points)")
axes[0].set_title("All Models — Test MAE\n(green = best)")
axes[0].legend(fontsize=8)
axes[0].invert_yaxis()

# (b) Ridge coefficients
top_coef = coef_df.head(13)
bar_colors = ["#e74c3c" if v < 0 else "#3498db" for v in top_coef.values]
axes[1].barh(top_coef.index[::-1], top_coef.values[::-1], color=bar_colors[::-1])
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("Coefficient (standardized)")
axes[1].set_title("Ridge Coefficients\n(blue=positive, red=negative)")

# (c) Predicted vs Actual
best_preds_data = next(r["preds"] for r in results if r["model"] == best_name)
y_te_plot = test.dropna(subset=FEATURES_LASSO + ["Total"])["Total"].values
if len(best_preds_data) == len(y_te_plot):
    axes[2].scatter(y_te_plot, best_preds_data, alpha=0.5, s=18, color="#2ecc71")
    mn = min(y_te_plot.min(), best_preds_data.min()) - 5
    mx = max(y_te_plot.max(), best_preds_data.max()) + 5
    axes[2].plot([mn,mx],[mn,mx],"r--",lw=1)
    axes[2].set_xlabel("Actual Total")
    axes[2].set_ylabel("Predicted Total")
    axes[2].set_title(f"Predicted vs Actual\n{best_name}\nMAE={best['test_mae']:.2f}")

plt.tight_layout()
plt.savefig("total_model_v3_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: total_model_v3_results.png")

# ─────────────────────────────────────────────
# 12. SAVE FITTED MODEL ARTIFACTS
# ─────────────────────────────────────────────
# Re-fit on ALL available data for use in generate_predictions.py
print("\n── Fitting final model on full dataset ──")
full_train = df.dropna(subset=FEATURES_LASSO + ["Total"]).copy()
X_full = full_train[FEATURES_LASSO].values.astype(float)
y_full = full_train["Total"].values

sc_final    = StandardScaler()
X_full_sc   = sc_final.fit_transform(X_full)
model_final = Ridge(alpha=500).fit(X_full_sc, y_full)

print(f"  Full training games: {len(full_train)}")
print(f"  Intercept: {model_final.intercept_:.2f}")
print(f"  Coefficients: {dict(zip(FEATURES_LASSO, model_final.coef_.round(4)))}")
print("\nDone. Use model_final + sc_final + FEATURES_LASSO in generate_predictions.py")