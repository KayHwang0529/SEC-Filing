from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config
from features import BASELINE_COLUMNS, SIGNAL_COLUMNS

CONTROLS = ["pre_ret_60", "pre_vol_60"]


# ----------------------------------------------------------------------------- splits
def split_train_test(df: pd.DataFrame, test_start: str = config.TEST_START,
                     embargo_days: int = config.EMBARGO_DAYS):
    """Train = events whose full return window closes before test_start."""
    t = pd.Timestamp(test_start)
    cutoff = t - pd.tseries.offsets.BDay(embargo_days)
    train = df[df["event_day"] < cutoff].copy()
    test = df[df["event_day"] >= t].copy()
    return train, test


def walk_forward_splits(df: pd.DataFrame, first_test_year: int, embargo_days: int = config.EMBARGO_DAYS):
    """Expanding-window yearly folds: train on everything before year Y (minus embargo), test on Y."""
    for year in range(first_test_year, df["event_day"].dt.year.max() + 1):
        tr, te = split_train_test(df, f"{year}-01-01", embargo_days)
        te = te[te["event_day"].dt.year == year]
        if len(tr) and len(te):
            yield year, tr, te


# ----------------------------------------------------------------------------- preprocessing
class Preprocessor:
    """Winsorize + standardize, with parameters learned on train only."""

    def __init__(self, cols: list[str], q: float = 0.01):
        self.cols, self.q = cols, q

    def fit(self, df: pd.DataFrame):
        x = df[self.cols]
        self.lo, self.hi = x.quantile(self.q), x.quantile(1 - self.q)
        clipped = x.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = clipped.mean(), clipped.std().replace(0, 1)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.cols] = ((df[self.cols].clip(self.lo, self.hi, axis=1) - self.mu) / self.sd)
        return out


def _quarter(df: pd.DataFrame) -> pd.Series:
    return df["event_day"].dt.to_period("Q").astype(str)


# ----------------------------------------------------------------------------- univariate: IC
def ic_by_period(df: pd.DataFrame, signal: str, target: str, min_names: int = 5) -> pd.Series:
    """Cross-sectional Spearman IC per quarter. Needs >= min_names firms per quarter --
    with a handful of tickers, use pooled_rank_corr instead."""
    out = {}
    for q, g in df.dropna(subset=[signal, target]).groupby(_quarter(df)):
        if len(g) >= min_names:
            out[q] = stats.spearmanr(g[signal], g[target]).statistic
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 4:
        return {"ic_mean": np.nan, "ic_t_nw": np.nan, "ic_p": np.nan, "n_periods": len(ic)}
    res = sm.OLS(ic.values, np.ones(len(ic))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {"ic_mean": ic.mean(), "ic_t_nw": res.tvalues[0], "ic_p": res.pvalues[0], "n_periods": len(ic)}


def pooled_rank_corr(df: pd.DataFrame, signal: str, target: str) -> dict:
    d = df.dropna(subset=[signal, target])
    if len(d) < 10:
        return {"rho": np.nan, "rho_p": np.nan, "n": len(d)}
    r = stats.spearmanr(d[signal], d[target])
    return {"rho": r.statistic, "rho_p": r.pvalue, "n": len(d)}


def permutation_pvalue(df: pd.DataFrame, signal: str, target: str, n_perm: int = 2000,
                       seed: int = 0) -> float:
    """Two-sided p-value for pooled Spearman rho, shuffling the signal WITHIN quarter
    (preserves each quarter's market-wide return level)."""
    d = df.dropna(subset=[signal, target]).copy()
    if len(d) < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    q = _quarter(d).values
    y = stats.rankdata(d[target])
    x = d[signal].values
    obs = abs(stats.spearmanr(x, y).statistic)
    if not np.isfinite(obs):  # constant signal -> undefined correlation
        return np.nan
    groups = [np.where(q == g)[0] for g in np.unique(q)]
    hits = 0
    for _ in range(n_perm):
        xp = x.copy()
        for idx in groups:
            xp[idx] = rng.permutation(xp[idx])
        hits += abs(stats.spearmanr(xp, y).statistic) >= obs
    return (hits + 1) / (n_perm + 1)


# ----------------------------------------------------------------------------- multivariate
def fit_ols(df: pd.DataFrame, y: str, X: list[str]):
    """OLS with two-way clustered SEs (firm, quarter)."""
    d = df.dropna(subset=[y] + X)
    groups = np.column_stack([pd.factorize(d["ticker"])[0], pd.factorize(_quarter(d))[0]])
    model = sm.OLS(d[y], sm.add_constant(d[X], has_constant="add"))
    try:
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:  # too few clusters -> fall back to heteroskedasticity-robust
        return model.fit(cov_type="HC3")


def oos_metrics(train: pd.DataFrame, test: pd.DataFrame, y: str, X: list[str]) -> dict:
    """Fit on train, score on test. R2_oos is vs the TRAIN-mean forecast (Campbell-Thompson);
    positive = beats naive. Also directional hit rate and top-minus-bottom quintile spread."""
    tr = train.dropna(subset=[y] + X)
    te = test.dropna(subset=[y] + X)
    if len(tr) < 30 or len(te) < 10:
        return {"n_train": len(tr), "n_test": len(te)}
    fit = sm.OLS(tr[y], sm.add_constant(tr[X], has_constant="add")).fit()
    pred = fit.predict(sm.add_constant(te[X], has_constant="add"))
    err_model = ((te[y] - pred) ** 2).sum()
    err_naive = ((te[y] - tr[y].mean()) ** 2).sum()
    hits = int(((pred > 0) == (te[y] > 0)).sum())
    q = pd.qcut(pred.rank(method="first"), 5, labels=False)
    spread = te[y][q == 4].mean() - te[y][q == 0].mean()
    spread_t = stats.ttest_ind(te[y][q == 4], te[y][q == 0], equal_var=False)
    return {
        "n_train": len(tr), "n_test": len(te),
        "r2_oos": 1 - err_model / err_naive,
        "hit_rate": hits / len(te),
        "hit_p": stats.binomtest(hits, len(te), 0.5).pvalue,
        "q5_minus_q1": spread, "q5_minus_q1_t": spread_t.statistic,
    }


# ----------------------------------------------------------------------------- study
def run_study(df: pd.DataFrame, signals=SIGNAL_COLUMNS, baselines=BASELINE_COLUMNS,
              horizons=config.HORIZONS, windows=("car_0", "car_1"), n_perm: int = 1000) -> dict:
    df = df.copy()
    df["event_day"] = pd.to_datetime(df["event_day"])
    usable = lambda s: s in df and df[s].notna().sum() > 20 and df[s].nunique() > 1
    signals = [s for s in signals if usable(s)]
    baselines = [s for s in baselines if usable(s)]
    controls = [c for c in CONTROLS if c in df]
    feats = signals + baselines + controls

    train, test = split_train_test(df)
    print(f"train: {len(train)} events  ({train['event_day'].min():%Y-%m} .. {train['event_day'].max():%Y-%m})")
    print(f"test : {len(test)} events  ({test['event_day'].min():%Y-%m} .. {test['event_day'].max():%Y-%m})")
    pp = Preprocessor(feats).fit(train)
    train, test = pp.transform(train), pp.transform(test)

    # 1) Univariate screen: each signal x horizon, evaluated separately on train and test.
    uni = []
    for w in windows:
        for h in horizons:
            y = f"{w}_{h}"
            if y not in df:
                continue
            for s in signals + baselines:
                row = {"signal": s, "target": y, "is_baseline": s in baselines}
                for name, part in (("train", train), ("test", test)):
                    pr = pooled_rank_corr(part, s, y)
                    ic = ic_summary(ic_by_period(part, s, y))
                    row |= {f"{name}_{k}": v for k, v in (pr | ic).items()}
                row["test_perm_p"] = permutation_pvalue(test, s, y, n_perm=n_perm)
                uni.append(row)
    uni = pd.DataFrame(uni)
    if len(uni):
        # FDR control across EVERY test you ran on the hold-out set
        ok = uni["test_perm_p"].notna()
        uni.loc[ok, "test_q_bh"] = multipletests(uni.loc[ok, "test_perm_p"], method="fdr_bh")[1]

    # 2) Nested model comparison out of sample: controls vs +baseline vs +LLM vs all.
    specs = {"controls": controls, "baseline": controls + baselines,
             "llm": controls + signals, "llm+baseline": controls + signals + baselines}
    oos = []
    for w in windows:
        for h in horizons:
            y = f"{w}_{h}"
            if y not in df:
                continue
            for name, X in specs.items():
                if X:
                    oos.append({"target": y, "model": name} | oos_metrics(train, test, y, X))
    oos = pd.DataFrame(oos)

    # 3) In-sample regression on TRAIN for interpretation (coefficients, clustered SEs).
    ins = {}
    for w in windows:
        for h in horizons:
            y = f"{w}_{h}"
            if y in df and signals:
                ins[y] = fit_ols(train, y, controls + signals)

    return {"univariate": uni, "oos": oos, "in_sample": ins, "train": train, "test": test}


def save_results(res: dict, out_dir=config.RESULTS_DIR):
    res["univariate"].to_csv(out_dir / "univariate.csv", index=False)
    res["oos"].to_csv(out_dir / "oos_models.csv", index=False)
    with open(out_dir / "in_sample_ols.txt", "w") as f:
        for y, fit in res["in_sample"].items():
            f.write(f"\n{'=' * 30} {y} {'=' * 30}\n{fit.summary()}\n")
    print(f"results -> {out_dir}")
