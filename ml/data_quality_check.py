"""
Day 1 — data quality check for FraudGuard.
"""

import sys
from pathlib import Path

import pandas as pd


def run(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    print("=== Shape ===")
    print(df.shape)

    print("\n=== Dtypes ===")
    print(df.dtypes.value_counts())

    print("\n=== Missing values ===")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.any() else "None — clean.")

    print("\n=== Duplicate rows ===")
    dupes = df.duplicated().sum()
    print(f"{dupes} duplicate rows ({dupes / len(df):.3%})")

    print("\n=== Class balance (Class: 1 = fraud) ===")
    counts = df["Class"].value_counts()
    pct = df["Class"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "pct": pct.round(4)}))
    print(
        "\n^ This is the imbalance you'll design around from Day 3 onward — "
        "plain accuracy will be meaningless on this split."
    )

    print("\n=== Amount summary (fraud vs legit) ===")
    print(df.groupby("Class")["Amount"].describe())

    print("\n=== Time range ===")
    print(f"Time spans {df['Time'].min():.0f}s to {df['Time'].max():.0f}s "
          f"({(df['Time'].max() - df['Time'].min()) / 3600:.1f} hours of data)")

    print("\n=== Leakage sanity check ===")
    print(
        "V1-V28 are pre-computed PCA components from the original raw "
        "features (already anonymized by the dataset publisher) — Time and "
        "Amount are the only two you can engineer further yourself. There is "
        "no obviously fraud-derived column here (e.g. no 'flagged_by_bank' "
        "field), but confirm this holds before trusting it blindly."
    )

    dupe_frauds = df[df.duplicated()]["Class"].sum()
    print(f"\nDuplicate rows that are fraud: {dupe_frauds} "
          f"(decide whether to drop duplicates before or after train/test split)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_quality_check.py <path-to-creditcard.csv>")
        sys.exit(1)
    run(Path(sys.argv[1]))