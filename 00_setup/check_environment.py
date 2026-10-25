"""Confirm the environment is ready.

Prints the Python and package versions, flags anything missing, then actually trains a
small model - importing a library is not the same as the library working.

    python 00_setup/check_environment.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import platform
import sys

# (import name, distribution name, which part needs it)
CORE = [
    ("numpy", "numpy", "everything"),
    ("pandas", "pandas", "everything"),
    ("scipy", "scipy", "everything"),
    ("sklearn", "scikit-learn", "parts 01-03"),
    ("matplotlib", "matplotlib", "everything"),
    ("seaborn", "seaborn", "plots"),
    ("statsmodels", "statsmodels", "time series"),
    ("xgboost", "xgboost", "gradient boosting"),
    ("requests", "requests", "dataset downloads"),
]

DEEP_LEARNING = [
    ("tensorflow", "tensorflow", "part 04, part 05"),
    ("torch", "torch", "part 04"),
    ("jax", "jax", "part 04"),
    ("google.cloud.aiplatform", "google-cloud-aiplatform", "part 05"),
    ("google.cloud.bigquery", "google-cloud-bigquery", "part 05"),
    ("kfp", "kfp", "part 05 pipelines"),
]

GREEN, YELLOW, RESET = "", "", ""


def rule(title: str = "") -> None:
    print(f"\n{title}\n" + "-" * 68 if title else "-" * 68)


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    print(f"  Python        {platform.python_version()}  ({platform.system()} {platform.machine()})")
    print(f"  Executable    {sys.executable}")

    if (major, minor) == (3, 12):
        print("  Version       OK - 3.12 is the version this repository targets")
        return True
    if (major, minor) < (3, 9):
        print(f"  Version       TOO OLD - 3.12 recommended, you have {major}.{minor}")
        return False
    if (major, minor) > (3, 12):
        print(f"  Version       {major}.{minor} - fine for parts 01-03.")
        print("                TensorFlow has no wheel above 3.12, so parts 04-05 will")
        print("                fail to install. See SETUP.md.")
        return True
    print(f"  Version       {major}.{minor} - workable, though 3.12 is the target")
    return True


def check_group(group: list[tuple[str, str, str]], label: str) -> list[str]:
    rule(label)
    missing: list[str] = []
    for import_name, dist_name, needed_for in group:
        try:
            importlib.import_module(import_name)
            try:
                version = metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                version = "?"
            print(f"  [ok]      {dist_name:<28} {version:<12} {needed_for}")
        except ImportError:
            print(f"  [MISSING] {dist_name:<28} {'-':<12} {needed_for}")
            missing.append(dist_name)
    return missing


def smoke_test() -> bool:
    """Train something real. Imports succeeding does not prove the stack works."""
    rule("Smoke test - train a model end to end")
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        print(f"  Loaded Iris                 {X.shape[0]} rows x {X.shape[1]} features")
        print(f"  Trained RandomForest        50 trees")
        print(f"  Test accuracy               {accuracy:.3f}")

        if accuracy < 0.80:
            print("  [WARN] Accuracy is implausibly low for Iris - something is wrong.")
            return False
        print("  [ok] scikit-learn is working.")
        return True
    except Exception as exc:  # noqa: BLE001 - we want any failure reported, not raised
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("=" * 68)
    print("  ENVIRONMENT CHECK")
    print("=" * 68)

    rule("Interpreter")
    python_ok = check_python()

    missing_core = check_group(CORE, "Core packages  (parts 01-03)")
    missing_dl = check_group(DEEP_LEARNING, "Deep learning and cloud  (parts 04-05)")

    smoke_ok = smoke_test() if not missing_core else False

    rule("Summary")
    if missing_core:
        print(f"  Core packages missing: {', '.join(missing_core)}")
        print("    ->  pip install -r requirements.txt")
    else:
        print("  Core packages      all present")

    if missing_dl:
        print(f"  Deep learning missing: {', '.join(missing_dl)}")
        print("    ->  pip install -r requirements-dl.txt   (only needed from part 04)")
    else:
        print("  Deep learning      all present")

    ready = python_ok and not missing_core and smoke_ok
    print()
    if ready:
        print("  READY. Next:  python 00_setup/download_datasets.py")
    else:
        print("  NOT READY - resolve the items above, then re-run this script.")
    print("=" * 68)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
