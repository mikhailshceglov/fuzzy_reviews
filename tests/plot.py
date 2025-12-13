import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def group_metric(rows: List[dict], key: str) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for r in rows:
        mf = int(r["min_freq"])
        out.setdefault(mf, []).append(float(r[key]))
    return out


def mean_std(x: List[float]) -> Tuple[float, float]:
    a = np.array(x, dtype=np.float64)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def auto_ylim(values: List[float], pad: float = 0.01) -> Tuple[float, float]:
    v = np.array(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    lo = float(v.min()) - pad
    hi = float(v.max()) + pad
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    if hi - lo < 0.02:
        mid = 0.5 * (hi + lo)
        lo = max(0.0, mid - 0.02)
        hi = min(1.0, mid + 0.02)
    return lo, hi


def ecdf(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(a)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_01_boxplot_rho(rho_by_mf: Dict[int, List[float]], out_png: Path) -> None:
    mfs = sorted(rho_by_mf.keys())
    data = [rho_by_mf[mf] for mf in mfs]
    all_rho = [v for mf in mfs for v in rho_by_mf[mf]]
    y0, y1 = auto_ylim(all_rho, pad=0.003)

    plt.figure(figsize=(8.0, 7.5))
     
    plt.boxplot(
        data,
        labels=[str(mf) for mf in mfs],
        showfliers=False,
        widths=0.55,   # <-- коробки шире (не выглядят "сжатыми")
    )

    rng = np.random.default_rng(42)
    for i, mf in enumerate(mfs, start=1):
        y = np.array(rho_by_mf[mf], dtype=np.float64)
        if len(y) == 0:
            continue
        x = rng.normal(loc=i, scale=0.06, size=len(y))  # <-- чуть больше джиттер
        plt.plot(x, y, marker="o", linestyle="None", markersize=7, alpha=0.85)

    plt.ylim(y0, y1)
    plt.grid(True, linewidth=0.6, alpha=0.35)
    plt.xlabel("min_freq")
    plt.ylabel(r"Spearman $\rho$")
    plt.title(r"Lexicon stability across repeats ($\rho$ by min_freq)")  # <-- без \_
    plt.tight_layout(pad=1.2)
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_02_tradeoff_rho_intersection(
    rho_by_mf: Dict[int, List[float]],
    inter_by_mf: Dict[int, List[float]],
    out_png: Path,
) -> None:
    mfs = sorted(rho_by_mf.keys())
    rho_means, rho_stds = [], []
    inter_means = []

    all_rho = []
    for mf in mfs:
        m, s = mean_std(rho_by_mf[mf])
        rho_means.append(m)
        rho_stds.append(s)
        all_rho.extend(rho_by_mf[mf])

        im, _ = mean_std(inter_by_mf.get(mf, []))
        inter_means.append(im)

    y0, y1 = auto_ylim(all_rho, pad=0.01)

    fig = plt.figure()
    ax1 = fig.gca()

    ax1.errorbar(mfs, rho_means, yerr=rho_stds, marker="o", linestyle="-", capsize=4)
    ax1.set_xlabel("min_freq")
    ax1.set_ylabel(r"Spearman $\rho$ (mean ± std)")
    ax1.set_ylim(y0, y1)
    ax1.set_xticks(mfs)
    ax1.grid(True, linewidth=0.5, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(mfs, inter_means, marker="o", linestyle="--")
    ax2.set_ylabel("Intersection size (mean)")

    ax1.set_title(r"Trade-off: stability ($\rho$) vs coverage (intersection)")
    fig.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_03_scatter_pairs(npz_path: Path, min_freq: int, out_png: Path) -> None:
    z = np.load(npz_path, allow_pickle=True)
    x = z.get(f"mf{min_freq}_x", None)
    y = z.get(f"mf{min_freq}_y", None)
    if x is None or y is None or len(x) == 0:
        raise SystemExit(f"В npz нет данных для scatter min_freq={min_freq}.")

    plt.figure()
    plt.plot([-1, 1], [-1, 1], linestyle="--")
    plt.scatter(x, y, s=10, alpha=0.6)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, linewidth=0.5, alpha=0.35)
    plt.xlabel("polarity A")
    plt.ylabel("polarity B")
    plt.title(f"pol_A(w) vs pol_B(w) (sample), min_freq={min_freq}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_04_ecdf_deltas(npz_path: Path, min_freqs: List[int], out_png: Path) -> None:
    z = np.load(npz_path, allow_pickle=True)

    all_d = []
    d_by_mf = {}
    for mf in min_freqs:
        d = z.get(f"mf{mf}_d", None)
        if d is None or len(d) == 0:
            continue
        d = d.astype(np.float64)
        d = d[np.isfinite(d)]
        d_by_mf[mf] = d
        all_d.append(d)

    if not all_d:
        raise SystemExit("В npz нет данных по |Δ|.")

    all_d = np.concatenate(all_d, axis=0)
    x_max = float(np.quantile(all_d, 0.99)) * 1.05
    x_max = min(0.6, max(0.12, x_max))

    plt.figure()
    for mf in min_freqs:
        d = d_by_mf.get(mf)
        if d is None:
            continue
        x, y = ecdf(np.sort(d))
        plt.plot(x, y, label=f"min_freq={mf}")

    plt.xlim(0.0, x_max)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linewidth=0.5, alpha=0.35)
    plt.xlabel(r"$|pol_A(w) - pol_B(w)|$")
    plt.ylabel("ECDF")
    plt.title("ECDF of |Δ| (zoomed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=Path("tests/stability_report.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("tests/stability_pairs_rep0.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("tests/plots"))
    ap.add_argument("--scatter-min-freq", type=int, default=50)
    args = ap.parse_args()

    rows = load_jsonl(args.report)
    rho_by_mf = group_metric(rows, "rho_spearman")
    inter_by_mf = group_metric(rows, "intersection_size")
    mfs = sorted(rho_by_mf.keys())

    ensure_dir(args.outdir)

    plot_01_boxplot_rho(rho_by_mf, args.outdir / "01_boxplot_rho_zoom.png")
    plot_02_tradeoff_rho_intersection(rho_by_mf, inter_by_mf, args.outdir / "02_tradeoff_rho_intersection.png")
    plot_03_scatter_pairs(args.pairs, args.scatter_min_freq, args.outdir / "03_scatter_polA_polB.png")
    plot_04_ecdf_deltas(args.pairs, mfs, args.outdir / "04_ecdf_abs_delta_zoom.png")

    print(f"[OK] Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
