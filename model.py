from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config
from features import BASELINE_COLUMNS, SIGNAL_COLUMNS

CONTROLS = ["pre_ret_60", "pre_vol_60", "earnings_car"]


# ----------------------------------------------------------------------------- splits
def split_train_test(
    df: pd.DataFrame,
    test_start: str = config.TEST_START,
    embargo_days: int = config.EMBARGO_DAYS,
):
    """Train = events whose full return window closes before test_start."""
    t = pd.Timestamp(test_start)
    cutoff = t - pd.tseries.offsets.BDay(embargo_days)
    train = df[df["event_day"] < cutoff].copy()
    test = df[df["event_day"] >= t].copy()
    return train, test


def walk_forward_splits(
    df: pd.DataFrame,
    first_test_year: int,
    embargo_days: int = config.EMBARGO_DAYS,
):
    """Expanding-window yearly folds: train on everything before year Y (minus embargo)."""
    for year in range(first_test_year, df["event_day"].dt.year.max() + 1):
        train, test = split_train_test(df, f"{year}-01-01", embargo_days)
        test = test[test["event_day"].dt.year == year]
        if len(train) and len(test):
            yield year, train, test


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
        out[self.cols] = (df[self.cols].clip(self.lo, self.hi, axis=1) - self.mu) / self.sd
        return out


def _preprocess_by_form(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each form's scaler on its training rows and apply it to that form only."""
    train_out, test_out = train.copy(), test.copy()
    forms = set(train["form"].dropna()) | set(test["form"].dropna())
    for form in forms:
        train_mask, test_mask = train["form"] == form, test["form"] == form
        if not train_mask.any():
            continue
        preprocessor = Preprocessor(cols).fit(train.loc[train_mask])
        train_out.loc[train_mask] = preprocessor.transform(train.loc[train_mask])
        test_out.loc[test_mask] = preprocessor.transform(test.loc[test_mask])
    return train_out, test_out


def _quarter(df: pd.DataFrame) -> pd.Series:
    return df["event_day"].dt.to_period("Q").astype(str)


# ----------------------------------------------------------------------------- univariate: IC
def ic_by_period(df: pd.DataFrame, signal: str, target: str, min_names: int = 5) -> pd.Series:
    """Cross-sectional Spearman IC per quarter."""
    data = df.dropna(subset=[signal, target])
    out = {}
    for quarter, group in data.groupby(_quarter(data)):
        if len(group) >= min_names:
            out[quarter] = stats.spearmanr(group[signal], group[target]).statistic
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    """Only report Newey-West inference with at least 20 periods."""
    ic = ic.dropna()
    if len(ic) < 20:
        return {"ic_mean": ic.mean() if len(ic) else np.nan,
                "ic_t_nw": np.nan, "ic_p": np.nan, "n_periods": len(ic)}
    result = sm.OLS(ic.values, np.ones(len(ic))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    return {
        "ic_mean": ic.mean(),
        "ic_t_nw": result.tvalues[0],
        "ic_p": result.pvalues[0],
        "n_periods": len(ic),
    }


def pooled_rank_corr(df: pd.DataFrame, signal: str, target: str) -> dict:
    data = df.dropna(subset=[signal, target])
    if len(data) < 10:
        return {"rho": np.nan, "rho_p": np.nan, "n": len(data)}
    result = stats.spearmanr(data[signal], data[target])
    return {"rho": result.statistic, "rho_p": result.pvalue, "n": len(data)}


def permutation_pvalue(
    df: pd.DataFrame, signal: str, target: str, n_perm: int = 2000, seed: int = 0
) -> float:
    """Two-sided pooled Spearman test, shuffling the firm-demeaned signal within quarter."""
    data = df.dropna(subset=[signal, target]).copy()
    if len(data) < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    quarter = _quarter(data).values
    target_rank = stats.rankdata(data[target])
    signal_values = data[signal].values
    observed = abs(stats.spearmanr(signal_values, target_rank).statistic)
    if not np.isfinite(observed):
        return np.nan
    groups = [np.where(quarter == value)[0] for value in np.unique(quarter)]
    hits = 0
    for _ in range(n_perm):
        permuted = signal_values.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        hits += abs(stats.spearmanr(permuted, target_rank).statistic) >= observed
    return (hits + 1) / (n_perm + 1)


# ----------------------------------------------------------------------------- multivariate
def fit_ols(df: pd.DataFrame, y: str, X: list[str]):
    """OLS with two-way clustered SEs (firm, quarter)."""
    data = df.dropna(subset=[y] + X)
    groups = np.column_stack([
        pd.factorize(data["ticker"])[0],
        pd.factorize(_quarter(data))[0],
    ])
    model = sm.OLS(data[y], sm.add_constant(data[X], has_constant="add"))
    try:
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:  # Too few clusters: use heteroskedasticity-robust inference.
        return model.fit(cov_type="HC3")


def oos_metrics(train: pd.DataFrame, test: pd.DataFrame, y: str, X: list[str]) -> dict:
    """Out-of-sample forecast measures; quintile spread SEs cluster by firm."""
    tr = train.dropna(subset=[y] + X)
    te = test.dropna(subset=[y] + X)
    if len(tr) < 30 or len(te) < 10:
        return {"n_train": len(tr), "n_test": len(te)}
    fit = sm.OLS(tr[y], sm.add_constant(tr[X], has_constant="add")).fit()
    pred = fit.predict(sm.add_constant(te[X], has_constant="add"))
    err_model = ((te[y] - pred) ** 2).sum()
    err_naive = ((te[y] - tr[y].mean()) ** 2).sum()
    hits = int(((pred > 0) == (te[y] > 0)).sum())
    quintile = pd.qcut(pred.rank(method="first"), 5, labels=False)
    extremes = te.loc[(quintile == 4) | (quintile == 0), [y, "ticker"]].copy()
    extremes["top_quintile"] = (quintile.loc[extremes.index] == 4).astype(float)
    spread = (
        extremes.loc[extremes["top_quintile"] == 1, y].mean()
        - extremes.loc[extremes["top_quintile"] == 0, y].mean()
    )
    if extremes["ticker"].nunique() >= 2:
        spread_fit = sm.OLS(
            extremes[y], sm.add_constant(extremes["top_quintile"], has_constant="add")
        ).fit(cov_type="cluster", cov_kwds={
            "groups": pd.factorize(extremes["ticker"])[0],
        })
        spread_t = spread_fit.tvalues["top_quintile"]
    else:
        spread_t = np.nan
    return {
        "n_train": len(tr),
        "n_test": len(te),
        "r2_oos": 1 - err_model / err_naive if err_naive else np.nan,
        "hit_rate": hits / len(te),
        "q5_minus_q1": spread,
        "q5_minus_q1_cluster_t": spread_t,
    }


# ----------------------------------------------------------------------------- study
def run_study(
    df: pd.DataFrame,
    signals=SIGNAL_COLUMNS,
    baselines=BASELINE_COLUMNS,
    horizons=config.HORIZONS,
    windows=("car_0", "car_1"),
    n_perm: int = 1000,
) -> dict:
    df = df.copy()
    df["event_day"] = pd.to_datetime(df["event_day"])
    if "earnings_car" not in df:
        raise ValueError(
            "earnings_car is required; run `python main.py build` with 8-K item 2.02 data"
        )
    if pd.Timestamp(config.TEST_START) < pd.Timestamp(config.MODEL_BOUNDARY_DATE):
        print(
            "WARNING: TEST_START precedes the configured model release boundary; "
            "this holdout may be contaminated."
        )
    usable = lambda signal: (
        signal in df and df[signal].notna().sum() > 20 and df[signal].nunique() > 1
    )
    signals = [signal for signal in signals if usable(signal)]
    baselines = [signal for signal in baselines if usable(signal)]
    controls = [control for control in CONTROLS if control in df]
    feats = signals + baselines + controls

    train, test = split_train_test(df)
    for name, part in (("train", train), ("test", test)):
        if len(part):
            print(
                f"{name}: {len(part)} events "
                f"({part['event_day'].min():%Y-%m} .. {part['event_day'].max():%Y-%m})"
            )
        else:
            print(f"{name}: 0 events")
    train, test = _preprocess_by_form(train, test, feats)

    primary_signal, primary_target = "sentiment_firm_demeaned", "car_1_20"
    primary_data = test[
        (test["form"] == "10-K")
        & test.get(primary_target, pd.Series(index=test.index, dtype=float)).notna()
    ].copy()
    primary = []
    primary_predictors = [primary_signal] + controls
    if (
        primary_signal in primary_data
        and primary_data[primary_predictors].notna().all(axis=1).sum() >= 10
    ):
        data = primary_data.dropna(subset=[primary_target] + primary_predictors)
        fit = fit_ols(data, primary_target, primary_predictors)
        primary.append({
            "form": "10-K",
            "signal": primary_signal,
            "target": primary_target,
            "n": len(data),
            "coefficient": fit.params.get(primary_signal, np.nan),
            "clustered_p": fit.pvalues.get(primary_signal, np.nan),
            "earnings_control_coefficient": fit.params.get("earnings_car", np.nan),
        })
    primary = pd.DataFrame(primary)

    # The single primary test is confirmatory; this broader signal grid is exploratory.
    univariate = []
    forms = sorted(df["form"].dropna().unique())
    for form in forms:
        form_train, form_test = train[train["form"] == form], test[test["form"] == form]
        for window in windows:
            for horizon in horizons:
                target = f"{window}_{horizon}"
                if target not in df:
                    continue
                for signal in signals + baselines:
                    row = {
                        "form": form,
                        "signal": signal,
                        "target": target,
                        "is_baseline": signal in baselines,
                        "is_primary": (
                            form == "10-K"
                            and signal == primary_signal
                            and target == primary_target
                        ),
                    }
                    for split_name, part in (
                        ("train", form_train), ("test", form_test)
                    ):
                        pooled = pooled_rank_corr(part, signal, target)
                        quarterly = ic_summary(ic_by_period(part, signal, target))
                        row |= {
                            f"{split_name}_{key}": value
                            for key, value in (pooled | quarterly).items()
                        }
                    row["test_perm_p"] = permutation_pvalue(
                        form_test, signal, target, n_perm=n_perm
                    )
                    univariate.append(row)
    univariate = pd.DataFrame(univariate)

    specs = {
        "controls": controls,
        "baseline": controls + baselines,
        "llm": controls + signals,
        "llm+baseline": controls + signals + baselines,
    }
    oos = []
    for form in forms:
        form_train, form_test = train[train["form"] == form], test[test["form"] == form]
        for window in windows:
            for horizon in horizons:
                target = f"{window}_{horizon}"
                if target not in df:
                    continue
                for name, predictors in specs.items():
                    if predictors:
                        oos.append({
                            "form": form,
                            "target": target,
                            "model": name,
                            **oos_metrics(form_train, form_test, target, predictors),
                        })
    oos = pd.DataFrame(oos)

    yearly = []
    first_test_year = max(int(df["event_day"].dt.year.min()) + 1, 2023)
    for year, fold_train, fold_test in walk_forward_splits(
        df, first_test_year, config.EMBARGO_DAYS
    ):
        fold_train, fold_test = _preprocess_by_form(fold_train, fold_test, feats)
        for form in forms:
            form_test = fold_test[fold_test["form"] == form]
            for window in windows:
                for horizon in horizons:
                    target = f"{window}_{horizon}"
                    if target not in df:
                        continue
                    for signal in signals + baselines:
                        yearly.append({
                            "form": form,
                            "year": year,
                            "signal": signal,
                            "target": target,
                            **pooled_rank_corr(form_test, signal, target),
                            **ic_summary(ic_by_period(form_test, signal, target)),
                        })
    yearly = pd.DataFrame(yearly)

    cutoff_comparison = []
    boundary = pd.Timestamp(config.MODEL_BOUNDARY_DATE)
    for form in forms:
        form_data = df[df["form"] == form]
        for period, part in (
            ("pre_boundary", form_data[form_data["event_day"] < boundary]),
            ("post_boundary", form_data[form_data["event_day"] >= boundary]),
        ):
            for window in windows:
                for horizon in horizons:
                    target = f"{window}_{horizon}"
                    if target not in df:
                        continue
                    for signal in signals + baselines:
                        cutoff_comparison.append({
                            "form": form,
                            "period": period,
                            "signal": signal,
                            "target": target,
                            **pooled_rank_corr(part, signal, target),
                            **ic_summary(ic_by_period(part, signal, target)),
                        })
    cutoff_comparison = pd.DataFrame(cutoff_comparison)

    in_sample = {}
    for form in forms:
        form_train = train[train["form"] == form]
        for window in windows:
            for horizon in horizons:
                target = f"{window}_{horizon}"
                if target in df and signals:
                    usable_rows = form_train.dropna(subset=[target] + controls + signals)
                    if len(usable_rows) >= 10:
                        in_sample[f"{form} {target}"] = fit_ols(
                            usable_rows, target, controls + signals
                        )

    return {
        "univariate": univariate,
        "oos": oos,
        "in_sample": in_sample,
        "primary": primary,
        "yearly_ic": yearly,
        "cutoff_comparison": cutoff_comparison,
        "train": train,
        "test": test,
    }


def save_results(res: dict, out_dir=config.RESULTS_DIR):
    res["univariate"].to_csv(out_dir / "univariate.csv", index=False)
    res["oos"].to_csv(out_dir / "oos_models.csv", index=False)
    res["primary"].to_csv(out_dir / "primary_hypothesis.csv", index=False)
    res["yearly_ic"].to_csv(out_dir / "walk_forward_ic.csv", index=False)
    res["cutoff_comparison"].to_csv(out_dir / "cutoff_comparison.csv", index=False)
    with open(out_dir / "in_sample_ols.txt", "w") as out:
        for target, fit in res["in_sample"].items():
            out.write(f"\n{'=' * 30} {target} {'=' * 30}\n{fit.summary()}\n")
    print(f"results -> {out_dir}")
