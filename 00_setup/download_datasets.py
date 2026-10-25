"""Fetch and cache every dataset that is not built into a library.

Idempotent: re-running skips anything already on disk. Nothing here is committed to git.

    python 00_setup/download_datasets.py
    python 00_setup/download_datasets.py --force     # re-fetch everything

Datasets built into scikit-learn (Iris, Wine, Digits, California Housing) and
Keras (MNIST, Fashion-MNIST) are NOT downloaded here - those libraries fetch and
cache them on first use.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Behind a corporate proxy that inspects TLS, Python's bundled certificate store does
# not contain the intercepting root certificate, so HTTPS fails with SSLError even
# though the browser works. truststore makes Python use the operating system's trust
# store instead, which does have it. Harmless everywhere else.
try:
    import truststore

    truststore.inject_into_ssl()
    _TRUSTSTORE = True
except ImportError:
    _TRUSTSTORE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

SEABORN_DATA = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master"
TIMEOUT = 60


@dataclass
class Dataset:
    key: str
    filename: str
    url: str
    used_by: str
    note: str = ""
    is_zip: bool = False
    zip_members: list[str] = field(default_factory=list)


UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
UCI_STATIC = "https://archive.ics.uci.edu/static/public"

# Every dataset here is deliberately neutral or positive in subject matter:
# flowers, grains, wine, penguins, bicycles, cars, banknotes, films. Learning
# material should not require you to spend a fortnight thinking about illness.
DATASETS: list[Dataset] = [
    Dataset(
        key="iris",
        filename="iris.csv",
        url=f"{SEABORN_DATA}/iris.csv",
        used_by="(none - loader only, no example uses it)",
        note="Also in sklearn.datasets.load_iris - the CSV is here for pandas work.",
    ),
    Dataset(
        key="penguins",
        filename="penguins.csv",
        url=f"{SEABORN_DATA}/penguins.csv",
        used_by="01_foundations/01_train_test_split_and_cv, 03_confusion_matrix_and_metrics, "
                "06_feature_engineering",
        note="Three species, real missing values. The messy-tabular example.",
    ),
    Dataset(
        key="flights",
        filename="flights.csv",
        url=f"{SEABORN_DATA}/flights.csv",
        used_by="01_foundations/01_train_test_split_and_cv, 02_traditional_ml/06_time_series, "
                "03_neural_networks/04_rnn_lstm_gru",
        note="Monthly airline passengers 1949-1960 - trend plus clear seasonality.",
    ),
    Dataset(
        key="rice",
        filename="Rice_Cammeo_Osmancik.arff",
        url=f"{UCI_STATIC}/545/rice+cammeo+and+osmancik.zip",
        used_by="01_foundations/03_confusion_matrix_and_metrics, 02_traditional_ml/02_classification, "
                "03_neural_networks/02_mlp_tabular",
        note="3,810 grains, 7 shape measurements, two varieties. Clean binary classification.",
        is_zip=True,
        zip_members=["Rice_Cammeo_Osmancik.arff"],
    ),
    Dataset(
        key="wine_white",
        filename="winequality-white.csv",
        url=f"{UCI}/wine-quality/winequality-white.csv",
        used_by="01_foundations/04_roc_vs_pr_curves",
        note="4,898 wines. quality>=7 is 21.6% - natural imbalance for the ROC vs PR demo.",
    ),
    Dataset(
        key="wine_red",
        filename="winequality-red.csv",
        url=f"{UCI}/wine-quality/winequality-red.csv",
        used_by="(none - loader only, no example uses it)",
        note="1,599 wines. Pairs with the white set.",
    ),
    Dataset(
        key="auto_mpg",
        filename="auto-mpg.data",
        url=f"{UCI}/auto-mpg/auto-mpg.data",
        used_by="01_foundations/02_bias_variance_tradeoff, 02_traditional_ml/01_regression",
        note="398 cars. Small enough to overfit visibly; 6 missing horsepower values.",
    ),
    Dataset(
        key="bike_day",
        filename="day.csv",
        url=f"{UCI_STATIC}/275/bike+sharing+dataset.zip",
        used_by="01_foundations/05_data_leakage, 06_feature_engineering, "
                "02_traditional_ml/05_anomaly_detection, 06_time_series",
        note="731 days / 17,379 hours of bike rentals. Regression, seasonality and anomalies.",
        is_zip=True,
        zip_members=["day.csv", "hour.csv", "Readme.txt"],
    ),
    Dataset(
        key="banknote",
        filename="banknote.csv",
        url=f"{UCI}/00267/data_banknote_authentication.txt",
        used_by="(none - loader only, no example uses it)",
        note="1,372 notes, 4 wavelet features, genuine vs forged. Very clean binary.",
    ),
    Dataset(
        key="seeds",
        filename="seeds.tsv",
        url=f"{UCI}/00236/seeds_dataset.txt",
        used_by="02_traditional_ml/03_clustering",
        note="210 wheat kernels, 3 varieties. An Iris-sized dataset with more features.",
    ),
    Dataset(
        key="adult",
        filename="adult.data",
        url=f"{UCI}/adult/adult.data",
        used_by="03_neural_networks/08_wide_and_deep, 07_serving/03_model_optimization, "
                "08_monitoring/01_skew_and_drift, 02_fairness_and_slices, 09_capstone",
        note="Census income. The dataset the Wide & Deep paper used.",
    ),
    Dataset(
        key="adult_names",
        filename="adult.names",
        url=f"{UCI}/adult/adult.names",
        used_by="(documentation only - load_adult() hardcodes the column names)",
        note="Column definitions for adult.data.",
    ),
    Dataset(
        key="movielens",
        filename="ml-100k/u.data",
        url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        used_by="02_traditional_ml/07_recommenders, 03_neural_networks/06_embeddings, "
                "09_two_tower_retrieval",
        note="100,000 ratings, 943 users, 1,682 movies. The standard recsys dataset.",
        is_zip=True,
        zip_members=["ml-100k/u.data", "ml-100k/u.item", "ml-100k/u.user", "ml-100k/README"],
    ),
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n/1:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def fetch(ds: Dataset, force: bool) -> tuple[str, str]:
    """Returns (status, detail). Never raises - a failure is reported, not fatal."""
    target = DATA_DIR / ds.filename

    if target.exists() and not force:
        return "cached", human(target.stat().st_size)

    try:
        response = requests.get(ds.url, timeout=TIMEOUT)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return "FAILED", f"{type(exc).__name__}"

    try:
        if ds.is_zip:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                wanted = ds.zip_members or archive.namelist()
                for member in wanted:
                    if member in archive.namelist():
                        out = DATA_DIR / member
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(archive.read(member))
            size = human(target.stat().st_size) if target.exists() else human(len(response.content))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            size = human(len(response.content))
    except Exception as exc:  # noqa: BLE001
        return "FAILED", f"{type(exc).__name__}"

    return "downloaded", size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / ".gitkeep").touch()

    print("=" * 74)
    print("  DATASET DOWNLOAD")
    print(f"  Target: {DATA_DIR}")
    print("=" * 74)

    results: list[tuple[str, str, str]] = []
    for ds in DATASETS:
        print(f"  {ds.key:<14} ... ", end="", flush=True)
        status, detail = fetch(ds, args.force)
        print(f"{status:<11} {detail}")
        results.append((ds.key, status, detail))

    failures = [key for key, status, _ in results if status == "FAILED"]

    print("-" * 74)
    ok = len(results) - len(failures)
    print(f"  {ok}/{len(results)} available")

    if failures and not _TRUSTSTORE:
        print()
        print("  truststore is not installed. If the failures below are SSLError, that")
        print("  is almost certainly a corporate proxy inspecting TLS. Fix:")
        print("      pip install truststore")
        print("  then re-run. It makes Python trust the same certificates your browser does.")

    if failures:
        print()
        print("  These failed - usually a proxy or a moved URL:")
        for ds in DATASETS:
            if ds.key in failures:
                print(f"    {ds.key}")
                print(f"      from {ds.url}")
                print(f"      to   {DATA_DIR / ds.filename}")
        print()
        print("  Fetch them by hand into the paths above, or re-run this script.")
        print("  Everything else still works - examples import only what they need.")

    print()
    print("  Library built-ins are NOT downloaded here - they fetch and cache")
    print("  themselves on first use, so duplicating them would waste bandwidth:")
    print("    scikit-learn  Iris, Wine, Digits, California Housing, make_blobs/moons")
    print("    Keras         MNIST, Fashion-MNIST   (part 03 onward)")
    print()
    print("  Next:  python 00_setup/loaders.py     # verify every loader parses")
    print("=" * 74)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
