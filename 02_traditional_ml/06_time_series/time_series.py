"""Time series: the one place where a random split is simply wrong.

  1. Decomposition - trend, seasonality and remainder, on the textbook series
  2. The baselines you must beat - naive and seasonal-naive are much harder
     to beat than people expect, and skipping them is how projects convince
     themselves they succeeded
  3. Random split vs chronological split - the same model, the same data,
     two completely different stories
  4. Lag features - how to turn forecasting into supervised regression
     without leaking the future
  5. Walk-forward validation, and the difference between a one-step-ahead
     error and a real multi-step forecast - only one of them compounds

Run:
    python 02_traditional_ml/06_time_series/time_series.py

Concept reference: Foundations volume, Parts A.7 (validation), A.10 (features).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_bike, load_flights  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit, train_test_split  # noqa: E402
from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


# ------------------------------------------------------- 1. decomposition
def figure_decomposition(flights: pd.DataFrame) -> None:
    """Every series is some combination of trend, a repeating seasonal shape,
    and what is left over. Seeing them separately tells you which model class
    you need before you fit anything.
    """
    s = flights.set_index("date")["passengers"]
    add = seasonal_decompose(s, model="additive", period=12)
    mul = seasonal_decompose(s, model="multiplicative", period=12)

    early = s[:24].std()
    late = s[-24:].std()
    print(f"  Monthly airline passengers, {s.index.min():%Y-%m} to {s.index.max():%Y-%m} "
          f"({len(s)} months).")
    print(f"  Level rises from {s[:12].mean():.0f} to {s[-12:].mean():.0f} per month.")
    print(f"\n  Standard deviation of the first 24 months:  {early:6.1f}")
    print(f"  Standard deviation of the last 24 months:   {late:6.1f}   "
          f"({late / early:.1f}x larger)")
    print("\n  The seasonal swings GROW with the level - the peaks get taller as")
    print("  the airline gets bigger. That makes the seasonality MULTIPLICATIVE,")
    print("  not additive, and it is the single most important thing to notice")
    print("  about this series before modelling it.")

    resid_add = np.nanstd(add.resid)
    resid_mul = np.nanstd(mul.resid * s.mean())
    print(f"\n  Residual spread, additive model:        {resid_add:6.1f}")
    print(f"  Residual spread, multiplicative model:  {resid_mul:6.1f}")
    better = "multiplicative" if resid_mul < resid_add else "additive"
    print(f"  The {better} decomposition leaves less unexplained.")
    print("\n  The standard fix is to model log(y): a multiplicative series becomes")
    print("  additive on the log scale, which is why so many forecasting recipes")
    print("  start with a log transform.")

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, (series, label) in zip(
        axes,
        [(s, "observed"), (add.trend, "trend"),
         (add.seasonal, "seasonal (additive)"), (add.resid, "remainder")],
    ):
        ax.plot(series.index, series.values, color=PALETTE["blue"], lw=1.4)
        ax.set_ylabel(label, fontsize=9.5)
    axes[3].axhline(0, color=PALETTE["grey"], lw=1)
    axes[3].set_xlabel("date")
    axes[0].set_title("Additive decomposition - note the remainder is NOT flat, "
                      "it fans out")
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '1_decomposition.png').name}")
    resid = add.resid.dropna().abs()
    by_year = resid.groupby(resid.index.year).mean()
    print("  Look at the remainder panel: it is U-shaped, not level. Mean")
    print("  |remainder| by year, additive model:")
    print("    " + "  ".join(f"{y}:{v:.0f}" for y, v in by_year.items()))
    print("  The additive model fits ONE seasonal amplitude - an average. That")
    print("  is too big for the small early years and too small for the large")
    print("  late ones, so the remainder is largest at both ends and smallest")
    print("  in the middle, where the level is closest to average. That is the")
    print("  additive model failing to absorb growing swings, visible as a")
    print("  shape rather than as a number.")


# ------------------------------------------------- 2. the baselines to beat
def figure_baselines(flights: pd.DataFrame) -> dict:
    """Naive and seasonal-naive cost one line each and are startlingly strong.
    Any forecasting result quoted without them is unevaluable.
    """
    s = flights.set_index("date")["passengers"]
    horizon = 24
    train, test = s[:-horizon], s[-horizon:]

    preds = {
        "Naive (last value)": np.repeat(train.iloc[-1], horizon),
        "Seasonal naive (value 12m ago)": train[-12:].to_numpy()[
            np.arange(horizon) % 12],
        "Mean of history": np.repeat(train.mean(), horizon),
        "Linear trend": None,   # filled below
    }
    x = np.arange(len(train))
    coef = np.polyfit(x, train.to_numpy(), 1)
    preds["Linear trend"] = np.polyval(coef, np.arange(len(train), len(train) + horizon))

    # drift: naive plus the average per-step change seen in training
    drift = train.iloc[-1] + (np.arange(1, horizon + 1)
                              * (train.iloc[-1] - train.iloc[0]) / (len(train) - 1))
    preds["Drift"] = drift

    print(f"  Holding out the last {horizon} months ({test.index.min():%Y-%m} "
          f"to {test.index.max():%Y-%m}).\n")
    print(f"  {'baseline':<32} {'MAE':>8} {'MAPE':>8}")
    results = {}
    for name, p in preds.items():
        p = np.asarray(p, dtype=float)
        results[name] = {"mae": mean_absolute_error(test, p),
                         "mape": mape(test.to_numpy(), p), "pred": p}
        print(f"  {name:<32} {results[name]['mae']:>8.1f} {results[name]['mape']:>7.1f}%")

    best = min(results, key=lambda k: results[k]["mae"])
    worst = max(results, key=lambda k: results[k]["mae"])
    print(f"\n  Best baseline: {best} (MAE {results[best]['mae']:.1f})")
    print(f"  Worst:         {worst} (MAE {results[worst]['mae']:.1f})")
    sn = results["Seasonal naive (value 12m ago)"]["mae"]
    print(f"\n  The spread between the best and worst one-line baseline is "
          f"{results[worst]['mae'] / results[best]['mae']:.1f}x.")
    print("  'Mean of history' is hopeless on a trending series - it is the")
    print("  regression baseline from 01_regression, and it does not transfer.")
    print(f"\n  Note which one actually won: {best}, at {results[best]['mae']:.1f} against")
    print(f"  seasonal naive's {sn:.1f}. This series is dominated by its TREND, so a")
    print("  straight line through 10 years of growth beats copying last year's")
    print("  shape. On a flat, strongly seasonal series the ranking reverses.")
    print("\n  That is the actual lesson: which baseline wins tells you what your")
    print("  series is made of. Run several, and let the winner point at the")
    print("  structure - here, trend first and seasonality second, which is")
    print("  exactly what the decomposition in section 1 showed.")
    print("\n  And whichever wins, REPORT IT NEXT TO YOUR MODEL. A forecasting")
    print("  result quoted without a baseline cannot be evaluated by anyone.")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(train.index, train.values, color=PALETTE["grey"], lw=1.3, label="train")
    ax.plot(test.index, test.values, color=PALETTE["black"], lw=2.2, label="actual")
    for (name, r), colour in zip(results.items(),
                                 ("blue", "vermillion", "green", "orange", "pink")):
        ax.plot(test.index, r["pred"], color=PALETTE[colour], lw=1.6, ls="--",
                label=f"{name} ({r['mae']:.0f})")
    ax.set_title("Five one-line baselines, and the last two years they are predicting")
    ax.set_xlabel("date"); ax.set_ylabel("passengers")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '2_baselines.png').name}")
    return results


# ------------------------------------- 3. the split that invents performance
def figure_split(bike: pd.DataFrame) -> None:
    """The headline mistake. A random split lets the model see the future, and
    on a smooth series it can then interpolate almost perfectly.
    """
    df = bike.copy()
    feats = ["temp", "atemp", "hum", "windspeed", "weathersit",
             "season", "holiday", "workingday", "mnth", "weekday"]
    X, y = df[feats].to_numpy(dtype=float), df["cnt"].to_numpy(dtype=float)

    # WRONG: rows shuffled, so training rows sit on both sides of every test row
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.25,
                                                  random_state=SEED)
    m_rand = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                   n_jobs=-1).fit(Xtr_r, ytr_r)
    mae_rand = mean_absolute_error(yte_r, m_rand.predict(Xte_r))

    # RIGHT: train on the past, test on the future
    cut = int(len(df) * 0.75)
    m_time = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                   n_jobs=-1).fit(X[:cut], y[:cut])
    pred_time = m_time.predict(X[cut:])
    mae_time = mean_absolute_error(y[cut:], pred_time)

    print(f"  Predicting daily bicycle rentals from weather and calendar only.\n")
    print(f"  Random 75/25 split       MAE {mae_rand:7.1f}   <- fiction")
    print(f"  Chronological 75/25      MAE {mae_time:7.1f}   <- honest")
    print(f"  The random split looks {mae_time / mae_rand:.1f}x better.")

    print("\n  Nothing is wrong with the model. The random split puts 2012-06-14")
    print("  in training and 2012-06-15 in test, and consecutive days are nearly")
    print("  identical - so the model is interpolating between known points")
    print("  rather than forecasting. At deployment there is no 'tomorrow' in")
    print(f"  your training set, and the error rises by {(mae_time / mae_rand - 1) * 100:.0f}%.")

    print("\n  There is a second effect here worth separating out. This series")
    print("  TRENDS UPWARD - 2012 is far busier than 2011 - so a chronological")
    print("  split also asks the model to extrapolate beyond any level it has")
    print("  seen. A tree CANNOT extrapolate: its predictions are averages of")
    print("  training values, so it is structurally capped at the 2011 maximum.")
    ytr_max, yte_max = y[:cut].max(), y[cut:].max()
    print(f"    highest value in training: {ytr_max:,.0f}")
    print(f"    highest value in test:     {yte_max:,.0f}")
    print(f"    highest value predicted:   {pred_time.max():,.0f}")
    print("  That third number can never exceed the first. Linear models and")
    print("  explicit trend terms can extrapolate; trees cannot, and this is the")
    print("  most common way a gradient-boosted forecast quietly under-predicts.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].scatter(yte_r, m_rand.predict(Xte_r), s=10, alpha=0.5,
                    color=PALETTE["vermillion"])
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], color=PALETTE["grey"], lw=1.2)
    axes[0].set_title(f"Random split - MAE {mae_rand:.0f}\n(the future leaked in)")
    axes[0].set_xlabel("actual"); axes[0].set_ylabel("predicted")

    axes[1].scatter(y[cut:], pred_time, s=10, alpha=0.5, color=PALETTE["blue"])
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], color=PALETTE["grey"], lw=1.2)
    axes[1].axhline(ytr_max, color=PALETTE["orange"], ls="--", lw=1.4)
    axes[1].annotate("training maximum - a tree\ncannot predict above this",
                     xy=(0.05, 0.80), xycoords="axes fraction", fontsize=9,
                     color=PALETTE["orange"], fontweight="bold")
    axes[1].set_title(f"Chronological split - MAE {mae_time:.0f}\n(honest)")
    axes[1].set_xlabel("actual"); axes[1].set_ylabel("predicted")

    fig.suptitle("The same model and the same data, split two ways",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    print(f"\n  saved  {save_figure(fig, OUT / '3_split_matters.png').name}")


# ---------------------------------------------------------- 4. lag features
def make_lags(df: pd.DataFrame, target: str, lags: tuple[int, ...],
              rolls: tuple[int, ...]) -> pd.DataFrame:
    """Turn a series into a supervised table.

    Every feature here uses .shift(), so row t only ever sees values from t-1
    or earlier. A rolling mean WITHOUT the shift would include today's value -
    the most common and most flattering time-series leak there is.
    """
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)
    for w in rolls:
        out[f"roll_mean_{w}"] = out[target].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[target].shift(1).rolling(w).std()
    return out.dropna().reset_index(drop=True)


def figure_lags(bike: pd.DataFrame) -> None:
    """Lag features let ordinary regressors forecast. The comparison that
    matters is against seasonal naive, not against zero.
    """
    base = ["temp", "atemp", "hum", "windspeed", "weathersit",
            "season", "holiday", "workingday", "mnth", "weekday"]
    df = make_lags(bike[["dteday", "cnt"] + base], "cnt",
                   lags=(1, 2, 3, 7, 14), rolls=(7, 28))
    feats = [c for c in df.columns if c not in ("dteday", "cnt")]
    X, y = df[feats].to_numpy(dtype=float), df["cnt"].to_numpy(dtype=float)
    cut = int(len(df) * 0.75)

    variants = {
        "Calendar + weather only": base,
        "Lags only": [c for c in feats if c.startswith(("lag_", "roll_"))],
        "Everything": feats,
    }
    print(f"  {len(df)} usable days after building lags "
          f"(the first 28 are lost to the longest window).\n")
    print(f"  {'features':<28} {'n':>4} {'MAE':>8}")

    results = {}
    for name, cols in variants.items():
        idx = [feats.index(c) for c in cols]
        m = RandomForestRegressor(n_estimators=300, random_state=SEED,
                                  n_jobs=-1).fit(X[:cut][:, idx], y[:cut])
        p = m.predict(X[cut:][:, idx])
        results[name] = {"mae": mean_absolute_error(y[cut:], p), "pred": p}
        print(f"  {name:<28} {len(cols):>4} {results[name]['mae']:>8.1f}")

    # seasonal naive on the same test window: last week's same weekday
    sn = df["cnt"].shift(7).to_numpy()[cut:]
    sn_mae = mean_absolute_error(y[cut:], sn)
    print(f"  {'seasonal naive (t-7)':<28} {'-':>4} {sn_mae:>8.1f}   <- the bar")

    best = min(results, key=lambda k: results[k]["mae"])
    print(f"\n  Best: {best} at MAE {results[best]['mae']:.1f}, against the seasonal-naive")
    print(f"  bar of {sn_mae:.1f}.")
    if results[best]["mae"] < sn_mae:
        print(f"  It beats the bar by {(1 - results[best]['mae'] / sn_mae) * 100:.0f}%. "
              f"That is the number to report -")
        print("  not the raw MAE, which means nothing on its own.")
    else:
        print("  It does NOT beat the bar. On this series the honest conclusion is")
        print("  that the model is not yet worth deploying over one line of pandas.")

    print("\n  Note how much the lag features add over calendar and weather alone.")
    print("  Yesterday's demand is the single most informative column you have")
    print("  in almost any demand-forecasting problem, and it is free.")
    print("\n  THE LEAK TO AVOID: every feature above is built with .shift() first.")
    print("  A rolling mean computed WITHOUT shifting includes today's value, so")
    print("  the model is handed a blurred copy of its own target. It is the most")
    print("  flattering bug in time series and it survives cross-validation,")
    print("  because the leak is inside the feature, not the split.")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    d = df["dteday"].iloc[cut:]
    ax.plot(d, y[cut:], color=PALETTE["black"], lw=2.0, label="actual")
    for (name, r), colour in zip(results.items(), ("orange", "green", "blue")):
        ax.plot(d, r["pred"], color=PALETTE[colour], lw=1.3,
                label=f"{name} (MAE {r['mae']:.0f})")
    ax.plot(d, sn, color=PALETTE["vermillion"], lw=1.1, ls=":",
            label=f"seasonal naive (MAE {sn_mae:.0f})")
    ax.set_title("Forecasting the held-out final quarter")
    ax.set_xlabel("date"); ax.set_ylabel("daily rentals")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '4_lag_features.png').name}")


# ------------------------------------------- 5. walk-forward and horizon
def figure_walkforward(bike: pd.DataFrame) -> None:
    """One chronological split is one sample. Walk-forward gives you several,
    and shows how the error behaves as you forecast further ahead.
    """
    base = ["temp", "hum", "windspeed", "weathersit", "workingday", "mnth"]
    df = make_lags(bike[["dteday", "cnt"] + base], "cnt",
                   lags=(1, 2, 7), rolls=(7,))
    feats = [c for c in df.columns if c not in ("dteday", "cnt")]
    X, y = df[feats].to_numpy(dtype=float), df["cnt"].to_numpy(dtype=float)

    tscv = TimeSeriesSplit(n_splits=5, test_size=60)
    print(f"  TimeSeriesSplit, 5 folds of 60 days. The training window GROWS and")
    print(f"  always precedes the test window - no fold ever sees its own future.\n")
    print(f"  {'fold':>5} {'train days':>11} {'test window':>26} {'MAE':>8} {'naive':>8}")
    maes, naives = [], []
    for i, (tr, te) in enumerate(tscv.split(X), start=1):
        m = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                  n_jobs=-1).fit(X[tr], y[tr])
        mae = mean_absolute_error(y[te], m.predict(X[te]))
        nv = mean_absolute_error(y[te], df["cnt"].shift(7).to_numpy()[te])
        maes.append(mae); naives.append(nv)
        w = f"{df['dteday'].iloc[te[0]]:%Y-%m-%d} to {df['dteday'].iloc[te[-1]]:%Y-%m-%d}"
        print(f"  {i:>5} {len(tr):>11} {w:>26} {mae:>8.1f} {nv:>8.1f}")

    print(f"\n  Mean MAE {np.mean(maes):.1f} +/- {np.std(maes):.1f} across folds, "
          f"against naive {np.mean(naives):.1f}.")
    print(f"  Fold-to-fold spread is {max(maes) - min(maes):.0f} MAE - a SINGLE split would")
    print("  have given you any one of these numbers and no idea of the range.")
    print("  That variation is seasonal: a winter fold and a summer fold are")
    print("  genuinely different problems.")

    # ---- horizon: one-step-ahead vs a genuine multi-step forecast
    #
    # These are two different questions and they are constantly conflated.
    # ONE-STEP-AHEAD re-reads the true value each day before predicting the
    # next, so the model never drifts. MULTI-STEP has to consume its own
    # predictions to build tomorrow's lags, so errors compound.
    cut = len(df) - 90
    m = RandomForestRegressor(n_estimators=300, random_state=SEED,
                              n_jobs=-1).fit(X[:cut], y[:cut])

    one_step = np.abs(y[cut:] - m.predict(X[cut:]))

    lag_cols = {c: i for i, c in enumerate(feats) if c.startswith(("lag_", "roll_"))}
    history = list(y[:cut])
    recursive_pred = []
    for t in range(cut, len(df)):
        row = X[t].copy()
        # overwrite every lag/rolling feature using the PREDICTED history
        for name, j in lag_cols.items():
            if name.startswith("lag_"):
                row[j] = history[-int(name.split("_")[1])]
            elif name.startswith("roll_mean_"):
                w = int(name.split("_")[2])
                row[j] = float(np.mean(history[-w:]))
            elif name.startswith("roll_std_"):
                w = int(name.split("_")[2])
                row[j] = float(np.std(history[-w:], ddof=1))
        p = float(m.predict(row.reshape(1, -1))[0])
        recursive_pred.append(p)
        history.append(p)            # the model now eats its own output
    recursive = np.abs(y[cut:] - np.array(recursive_pred))

    horizons = np.arange(1, 91)
    cum_one = np.array([one_step[:h].mean() for h in horizons])
    cum_rec = np.array([recursive[:h].mean() for h in horizons])

    print("\n  The horizon effect - but ONLY for a real multi-step forecast.")
    print(f"\n  {'horizon':>9} {'one-step-ahead':>16} {'recursive multi-step':>22}")
    for h in (1, 7, 30, 90):
        print(f"  {h:>6}d   {cum_one[h - 1]:>16.1f} {cum_rec[h - 1]:>22.1f}")

    print("\n  Read the two columns against each other. ONE-STEP-AHEAD error does")
    print("  NOT grow - it is flat or even falls, because every single day the")
    print("  model is handed yesterday's TRUE value before predicting today.")
    print("  That is a legitimate setup (it is what a daily-refreshed forecast")
    print("  does) but it is not a 90-day forecast, and quoting it as one is one")
    print("  of the most common overstatements in applied forecasting.")
    print(f"\n  RECURSIVE multi-step builds tomorrow's lags out of today's")
    print(f"  PREDICTION, so its errors compound: {cum_rec[6]:.0f} at 7 days rising to "
          f"{cum_rec[89]:.0f}")
    print(f"  at 90, while the one-step figure FALLS from {cum_one[6]:.0f} to {cum_one[89]:.0f}.")
    print(f"\n  Ignore the 1-day row - both columns are identical there (the first")
    print(f"  prediction still uses true lags) and it happens to be a badly missed")
    print("  day, which drags both cumulative means up. The signal is the GAP")
    print("  between the two columns, and it opens steadily:")
    for h in (7, 30, 60, 90):
        print(f"    at {h:>2} days the recursive forecast is "
              f"{cum_rec[h - 1] / cum_one[h - 1]:.2f}x the one-step error")
    print("\n  That widening ratio IS error compounding, isolated from everything")
    print("  else - same model, same data, same days, the only difference being")
    print("  whether the model was refed the truth each morning.")
    print("\n  ALWAYS quote the horizon AND which of the two you measured.")
    print("  'MAE 900' is not a claim on its own.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(np.arange(1, 6) - 0.2, maes, width=0.4,
                color=PALETTE["blue"], label="model")
    axes[0].bar(np.arange(1, 6) + 0.2, naives, width=0.4,
                color=PALETTE["vermillion"], label="seasonal naive")
    axes[0].set_xlabel("fold (chronological)"); axes[0].set_ylabel("MAE")
    axes[0].set_title("Walk-forward: five honest estimates, not one")
    axes[0].legend(fontsize=9)

    axes[1].plot(horizons, cum_one, color=PALETTE["blue"], lw=1.8,
                 label="one-step-ahead (true lags)")
    axes[1].plot(horizons, cum_rec, color=PALETTE["vermillion"], lw=1.8,
                 label="recursive multi-step")
    axes[1].set_xlabel("forecast horizon (days)")
    axes[1].set_ylabel("cumulative mean absolute error")
    axes[1].set_title("Error grows only when the model\neats its own predictions")
    axes[1].legend(fontsize=9)

    fig.suptitle("One split is one sample - and 'how far ahead' is two questions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '5_walkforward.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  TIME SERIES")
    print("=" * 72)

    flights = load_flights()
    bike = load_bike("day").sort_values("dteday").reset_index(drop=True)

    banner("1. Decomposition: trend, season, remainder")
    figure_decomposition(flights)

    banner("2. The baselines you have to beat")
    figure_baselines(flights)

    banner("3. Random split vs chronological split")
    figure_split(bike)

    banner("4. Lag features: forecasting as supervised regression")
    figure_lags(bike)

    banner("5. Walk-forward validation, and the horizon")
    figure_walkforward(bike)

    banner("What to take away")
    print("  - NEVER random-split a time series. It puts tomorrow in training")
    print("    and today in test, and the model interpolates instead of")
    print("    forecasting. Here it made the error look 2x better than it was.")
    print("  - Always report seasonal naive alongside your model. It is one")
    print("    line, it captures the whole repeating shape, and a large number")
    print("    of forecasting projects never actually beat it.")
    print("  - Build every lag and rolling feature with .shift() FIRST. An")
    print("    unshifted rolling mean contains today's target - a leak that")
    print("    lives inside the feature and survives any correct split.")
    print("  - TREES CANNOT EXTRAPOLATE. A prediction is an average of training")
    print("    values, so on a trending series a tree is capped at the highest")
    print("    level it has seen. Detrend first, or use a model with a trend.")
    print("  - Growing seasonal swings mean MULTIPLICATIVE seasonality; model")
    print("    log(y) and it becomes additive.")
    print("  - Use TimeSeriesSplit, not KFold. One chronological split is one")
    print("    sample, and the fold-to-fold spread here was large.")
    print("  - Quote the HORIZON and the REGIME with the error. One-step-ahead")
    print("    error stayed flat over 90 days here because the model was refed")
    print("    the truth daily; a genuine recursive forecast, eating its own")
    print("    predictions, ended up 1.5x worse over the same window.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
