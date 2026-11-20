#!/usr/bin/env python3
"""Generate the fixed calibration sample CSV (~200 households)."""

import csv
import random
from pathlib import Path

STREETS = ["河畔街道", "榆树街道", "望江街道", "古城街道", "新桥街道", "湖滨街道"]
TYPES = ["多层住宅", "小高层", "高层住宅", "联排住宅"]
TOTAL_TARGET = 200
SEED = 20261120


def main() -> None:
    rng = random.Random(SEED)
    rows = []

    # Give every street/type pair some households, then skew the rest.
    sample_no = 1
    for street in STREETS:
        for housing_type in TYPES:
            count = rng.choice([2, 2, 3])
            for _ in range(count):
                rows.append((f"HP-{sample_no:04d}", street, housing_type, rng.randint(2, 9)))
                sample_no += 1

    while len(rows) < TOTAL_TARGET:
        street = rng.choices(STREETS, weights=[2, 2, 3, 1, 2, 1])[0]
        housing_type = rng.choices(TYPES, weights=[3, 2, 2, 1])[0]
        rows.append((f"HP-{sample_no:04d}", street, housing_type, rng.randint(2, 9)))
        sample_no += 1

    rng.shuffle(rows)

    out = Path(__file__).resolve().parent.parent / "data" / "sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["样本号", "街道", "住房类型", "初始权重"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
