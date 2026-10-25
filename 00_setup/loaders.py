"""Clean loaders for every downloaded dataset.

Each function returns a tidy pandas DataFrame with real column names and correct
dtypes, and documents the dataset's quirks. Parsing lives here once so the example
scripts stay about the *concept* rather than about file formats.

Usage from any example script:

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "00_setup"))
    from loaders import load_rice

    df = load_rice()

Datasets built into a library are not here - call the library directly:

    from sklearn.datasets import load_iris, load_wine, load_digits, fetch_california_housing
    from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, imdb
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _require(relative: str) -> Path:
    path = DATA_DIR / relative
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing.\n"
            f"Run:  python 00_setup/download_datasets.py"
        )
    return path


# --------------------------------------------------------------------- tabular
def load_iris_df() -> pd.DataFrame:
    """150 x 5. Columns: sepal_length, sepal_width, petal_length, petal_width, species."""
    return pd.read_csv(_require("iris.csv"))


def load_penguins() -> pd.DataFrame:
    """344 x 7. Three species. Has genuine missing values - 2 rows lack all measurements
    and 11 lack sex. Deliberately left in: this is the messy-tabular example."""
    return pd.read_csv(_require("penguins.csv"))


def load_seeds() -> pd.DataFrame:
    """210 x 8. Seven geometric measurements of wheat kernels, three varieties
    (1=Kama, 2=Rosa, 3=Canadian). Whitespace separated with no header."""
    cols = [
        "area", "perimeter", "compactness", "kernel_length",
        "kernel_width", "asymmetry", "groove_length", "variety",
    ]
    df = pd.read_csv(_require("seeds.tsv"), sep=r"\s+", names=cols, engine="python")
    df["variety"] = df["variety"].map({1: "Kama", 2: "Rosa", 3: "Canadian"})
    return df


def load_rice() -> pd.DataFrame:
    """3,810 x 8. Seven shape measurements of rice grains, two varieties
    (Cammeo 1,630 / Osmancik 2,180). Stored as ARFF; scipy parses it."""
    from scipy.io import arff

    raw = _require("Rice_Cammeo_Osmancik.arff").read_text(encoding="utf-8", errors="replace")
    data, _meta = arff.loadarff(io.StringIO(raw))
    df = pd.DataFrame(data)
    # ARFF nominal fields arrive as bytes
    df["Class"] = df["Class"].str.decode("utf-8")
    df.columns = [c.lower() for c in df.columns]
    return df


def load_banknote() -> pd.DataFrame:
    """1,372 x 5. Four wavelet features of banknote photographs.
    class: 0 = genuine, 1 = forged. Almost perfectly separable - useful as the
    'what a clean problem looks like' baseline."""
    cols = ["variance", "skewness", "curtosis", "entropy", "class"]
    return pd.read_csv(_require("banknote.csv"), names=cols)


def load_auto_mpg() -> pd.DataFrame:
    """398 x 9. Fuel efficiency of 1970s-80s cars.
    Quirks, both deliberate teaching material:
      - horsepower has 6 missing values, encoded as '?'
      - car_name is quoted and contains spaces, so naive whitespace splitting fails
    """
    cols = [
        "mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name",
    ]
    # NOTE: do not pass engine="python" here. The python engine ignores quotechar
    # when the separator is a regex, which splits the quoted car_name on its spaces
    # and raises "Expected 9 fields, saw 12".
    return pd.read_csv(
        _require("auto-mpg.data"),
        sep=r"\s+",
        names=cols,
        na_values="?",
        quotechar='"',
    )


def load_wine_quality(colour: str = "white") -> pd.DataFrame:
    """4,898 white or 1,599 red wines x 12. Eleven physicochemical features and a
    quality score 0-10. Semicolon separated.

    Binarising with quality >= 7 gives 21.6% positives (white) - a natural,
    honest class imbalance for the ROC vs PR demonstration."""
    if colour not in {"white", "red"}:
        raise ValueError("colour must be 'white' or 'red'")
    df = pd.read_csv(_require(f"winequality-{colour}.csv"), sep=";")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def load_bike(granularity: str = "day") -> pd.DataFrame:
    """731 days or 17,379 hours of bicycle rentals, with weather and calendar
    features. 'cnt' is the target; 'casual' + 'registered' sum to it - which makes
    them a perfect target-leakage example."""
    if granularity not in {"day", "hour"}:
        raise ValueError("granularity must be 'day' or 'hour'")
    df = pd.read_csv(_require(f"{granularity}.csv"))
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df


def load_adult() -> pd.DataFrame:
    """32,561 x 15. Census income. Target 'income' is >50K for 24% of rows.
    Missing values are encoded as '?' with a leading space, hence skipinitialspace."""
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    return pd.read_csv(
        _require("adult.data"),
        names=cols,
        skipinitialspace=True,
        na_values="?",
    )


def load_flights() -> pd.DataFrame:
    """144 x 3. Monthly airline passengers 1949-1960. The textbook series with a
    clear upward trend and multiplicative seasonality."""
    df = pd.read_csv(_require("flights.csv"))
    # month is the full name ("January"), so %B - not %b, which expects "Jan".
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B"
    )
    return df.sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------- recommender
def load_movielens_ratings() -> pd.DataFrame:
    """100,000 x 4: user_id, item_id, rating (1-5), timestamp. Tab separated."""
    return pd.read_csv(
        _require("ml-100k/u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def load_movielens_movies() -> pd.DataFrame:
    """1,682 films: id, title, release date, and 19 binary genre flags.
    Pipe separated and latin-1 encoded - UTF-8 will fail on it."""
    genres = [
        "unknown", "action", "adventure", "animation", "childrens", "comedy",
        "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
        "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western",
    ]
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genres
    return pd.read_csv(
        _require("ml-100k/u.item"),
        sep="|",
        names=cols,
        encoding="latin-1",
    )


ALL_LOADERS = {
    "iris": load_iris_df,
    "penguins": load_penguins,
    "seeds": load_seeds,
    "rice": load_rice,
    "banknote": load_banknote,
    "auto_mpg": load_auto_mpg,
    "wine_quality": load_wine_quality,
    "bike": load_bike,
    "adult": load_adult,
    "flights": load_flights,
    "movielens_ratings": load_movielens_ratings,
    "movielens_movies": load_movielens_movies,
}


if __name__ == "__main__":
    print("Checking every loader\n" + "-" * 52)
    failures = 0
    for name, fn in ALL_LOADERS.items():
        try:
            df = fn()
            print(f"  [ok]   {name:<20} {df.shape[0]:>7,} x {df.shape[1]:>2}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name:<20} {type(exc).__name__}: {exc}")
            failures += 1
    print("-" * 52)
    print("  all loaders working" if not failures else f"  {failures} failed")
