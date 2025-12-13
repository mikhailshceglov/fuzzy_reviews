import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_lexicon"))

from build_auto_polarity_lexicon import SentimentLexiconBuilder


def read_reviews_lines(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def write_reviews(path: Path, reviews: List[str]) -> None:
    path.write_text("\n".join(reviews) + "\n", encoding="utf-8")


def rankdata(values: List[float]) -> List[float]:
    n = len(values)
    idx = list(range(n))
    idx.sort(key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[idx[i]]
        while j < n and values[idx[j]] == v:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[idx[k]] = avg_rank
        i = j
    return ranks


def pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman(x: List[float], y: List[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def apply_filters(
    lex: Dict[str, Dict[str, float]],
    min_freq: int,
    near_zero_eps: float,
    near_zero_max_freq: int,
) -> Dict[str, Tuple[float, int]]:
    out: Dict[str, Tuple[float, int]] = {}
    for w, rec in lex.items():
        p = float(rec["polarity"])
        f = int(rec["freq"])
        if f < min_freq:
            continue
        if near_zero_max_freq > 0 and abs(p) < near_zero_eps and f <= near_zero_max_freq:
            continue
        out[w] = (p, f)
    return out


def intersect_pairs(
    a: Dict[str, Tuple[float, int]],
    b: Dict[str, Tuple[float, int]],
) -> Tuple[List[float], List[float], List[float], int]:
    common = set(a.keys()) & set(b.keys())
    xa, xb, d = [], [], []
    for w in common:
        pa, _fa = a[w]
        pb, _fb = b[w]
        xa.append(pa)
        xb.append(pb)
        d.append(abs(pa - pb))
    return xa, xb, d, len(common)


def sample_pairs(
    rng: random.Random,
    xa: List[float],
    xb: List[float],
    d: List[float],
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(xa)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if n <= k:
        return np.array(xa, dtype=np.float32), np.array(xb, dtype=np.float32), np.array(d, dtype=np.float32)
    idx = rng.sample(range(n), k)
    return (
        np.array([xa[i] for i in idx], dtype=np.float32),
        np.array([xb[i] for i in idx], dtype=np.float32),
        np.array([d[i] for i in idx], dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews-txt", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, default=0.5)
    ap.add_argument("--min-freq", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--near-zero-eps", type=float, default=0.05)
    ap.add_argument("--near-zero-max-freq", type=int, default=50)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--model-name", type=str, default="cointegrated/rubert-tiny-sentiment-balanced")

    ap.add_argument("--pairs-repeat", type=int, default=0)
    ap.add_argument("--pairs-sample", type=int, default=5000)

    ap.add_argument("--json-out", type=Path, default=Path("tests/stability_report.jsonl"))
    ap.add_argument("--pairs-out", type=Path, default=Path("tests/stability_pairs_rep0.npz"))
    args = ap.parse_args()

    if args.repeats < 1 or args.repeats > 5:
        raise SystemExit("repeats должен быть в диапазоне 1..5 (по договорённости).")
    min_freqs = list(dict.fromkeys(args.min_freq))
    if min_freqs != [20, 50, 100]:
        print("[WARN] По договорённости min_freq=20/50/100. Сейчас:", min_freqs)

    reviews = read_reviews_lines(args.reviews_txt)
    if len(reviews) < 50:
        raise SystemExit("Слишком мало отзывов для оценки стабильности.")

    rng = random.Random(args.seed)

    builder = SentimentLexiconBuilder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_freq=1,
        near_zero_eps=args.near_zero_eps,
        near_zero_max_freq=0,
        encoding="utf-8",
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    jf = args.json_out.open("w", encoding="utf-8")

    pairs_payload: Dict[str, np.ndarray] = {}
    pairs_meta = {"pairs_repeat": args.pairs_repeat, "pairs_sample": args.pairs_sample, "min_freqs": min_freqs}

    for rep in range(args.repeats):
        idx = list(range(len(reviews)))
        rng.shuffle(idx)
        cut = int(len(idx) * args.split)
        a_reviews = [reviews[i] for i in idx[:cut]]
        b_reviews = [reviews[i] for i in idx[cut:]]

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pA = td / "A.txt"
            pB = td / "B.txt"
            write_reviews(pA, a_reviews)
            write_reviews(pB, b_reviews)

            builder.min_freq = 1
            builder.near_zero_max_freq = 0

            lex_a_raw = builder.build_lexicon([pA])
            lex_b_raw = builder.build_lexicon([pB])

            for mf in min_freqs:
                lex_a = apply_filters(lex_a_raw, mf, args.near_zero_eps, args.near_zero_max_freq)
                lex_b = apply_filters(lex_b_raw, mf, args.near_zero_eps, args.near_zero_max_freq)

                xa, xb, d, inter = intersect_pairs(lex_a, lex_b)
                rho = spearman(xa, xb)
                mad = float("nan") if inter == 0 else float(sum(d) / inter)

                rec = {
                    "repeat": rep,
                    "min_freq": mf,
                    "rho_spearman": rho,
                    "intersection_size": inter,
                    "mean_abs_diff": mad,
                    "lex_size_a": len(lex_a),
                    "lex_size_b": len(lex_b),
                    "n_reviews_a": len(a_reviews),
                    "n_reviews_b": len(b_reviews),
                    "near_zero_eps": args.near_zero_eps,
                    "near_zero_max_freq": args.near_zero_max_freq,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[rep {rep:03d}] min_freq={mf}  rho={rho:.4f}  inter={inter}  mad={mad:.4f}")

                if rep == args.pairs_repeat:
                    xs, ys, ds = sample_pairs(rng, xa, xb, d, args.pairs_sample)
                    pairs_payload[f"mf{mf}_x"] = xs
                    pairs_payload[f"mf{mf}_y"] = ys
                    pairs_payload[f"mf{mf}_d"] = ds

    jf.close()

    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.pairs_out, **pairs_payload, meta=json.dumps(pairs_meta, ensure_ascii=False))
    print(f"[OK] Saved: {args.json_out}")
    print(f"[OK] Saved: {args.pairs_out}")


if __name__ == "__main__":
    main()
