#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np


def summarize(arr: np.ndarray) -> str:
    return (
        f"shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():+.6f} max={arr.max():+.6f} "
        f"mean={arr.mean():+.6f} std={arr.std():+.6f}"
    )


def preview(arr: np.ndarray, n: int) -> str:
    flat = arr.flatten()
    head = ", ".join(f"{x:+.6f}" for x in flat[:n])
    tail = ", ".join(f"{x:+.6f}" for x in flat[-n:]) if len(flat) > n else ""
    if tail:
        return f"head[{n}]: {head}\ntail[{n}]: {tail}"
    return f"head[{n}]: {head}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="saved_weights", help="Directory with .npy files")
    ap.add_argument("--pattern", default="", help="Substring filter for filenames")
    ap.add_argument("--n", type=int, default=10, help="Preview count for head/tail")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Directory not found: {args.dir}")

    files: List[str] = [
        f for f in sorted(os.listdir(args.dir))
        if f.endswith(".npy") and (args.pattern in f)
    ]
    if not files:
        raise SystemExit("No .npy files found")

    for fname in files:
        path = os.path.join(args.dir, fname)
        arr = np.load(path)
        print(f"\n{fname}")
        print(summarize(arr))
        print(preview(arr, args.n))


if __name__ == "__main__":
    main()
