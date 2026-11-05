"""Create a smaller, class-stratified sample of a large CSV for presentations.

Example:
    python tools/make_sample.py combined_dataset.csv sample.csv --label label2 --per-class 5000
"""

import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--label", required=True, help="Name of the class-label column in your file")
    p.add_argument("--per-class", type=int, default=5000, help="Maximum rows kept per class")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    kept = {}
    for chunk in pd.read_csv(a.source, chunksize=a.chunksize, low_memory=False):
        if a.label not in chunk.columns:
            raise SystemExit(f"Column '{a.label}' not found. Available: {list(chunk.columns)[:20]} ...")
        for cls, part in chunk.groupby(a.label):
            cur = kept.get(cls)
            merged = part if cur is None else pd.concat([cur, part])
            kept[cls] = merged.sample(min(len(merged), a.per_class), random_state=a.seed)
    out = pd.concat(kept.values()).sample(frac=1, random_state=a.seed)
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out):,} rows to {a.output}")
    print(out[a.label].value_counts().to_string())


if __name__ == "__main__":
    main()
