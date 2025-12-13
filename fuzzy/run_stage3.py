from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Optional

from .engine import FuzzyEngine
from .io import read_jsonl, write_jsonl, find_by_id


def _fmt_float(x: float, nd: int = 3) -> str:
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _print_kv_block(title: str, items: Dict[str, Any], indent: int = 2) -> None:
    print(title)
    pad = " " * indent
    for k, v in items.items():
        if isinstance(v, float):
            vv = _fmt_float(v)
        else:
            vv = str(v)
        print(f"{pad}{k}: {vv}")


def _sorted_terms(term_mu: Dict[str, float]) -> list[tuple[str, float]]:
    return sorted(term_mu.items(), key=lambda kv: kv[1], reverse=True)


def _format_term_line(term_mu: Dict[str, float], top_k: int = 0, min_mu: float = 0.01) -> str:
    pairs = [(t, mu) for t, mu in _sorted_terms(term_mu) if mu >= min_mu]
    if top_k and top_k > 0:
        pairs = pairs[:top_k]
    if not pairs:
        return "∅"
    return " | ".join([f"{t} {_fmt_float(mu, 2)}" for t, mu in pairs])


def _pretty_report(
    *,
    engine: FuzzyEngine,
    features: Dict[str, Any],
    result: Any,
    show_text: bool,
    show_inputs_fuzzy: bool,
    show_derived: bool,
    top_k_terms: int,
    top_k_rules: int,
) -> None:
    line = "═" * 70
    review_id = features.get("id", "N/A")

    print(line)
    print("FUZZY REVIEWS — Stage 3 report (single review)")
    print(f"id: {review_id}")
    print(line)

    if show_text and isinstance(features.get("text"), str):
        print("TEXT")
        print(f'  "{features["text"]}"')
        print()

    # crisp + derived
    print("CRISP FEATURES (stage 2 + derived)")
    cf = result["crisp_features"]
    derived = result.get("derived_features", {})
    # Print selected crisp keys first (if present)
    preferred = [
        "score_crisp",
        "emotion_intensity_crisp",
        "coverage",
        "hedges_count",
        "tokens_alpha",
        "pos",
        "neg",
    ]
    printed = set()
    for k in preferred:
        if k in cf:
            printed.add(k)
            print(f"  {k:22s} {cf[k]}")
    # Print derived
    if show_derived:
        for k in ["magnitude", "conflict", "hedges_rate"]:
            if k in derived:
                print(f"  {k:22s} {_fmt_float(float(derived[k]), 3)}")
    # Print remaining crisp fields (compact)
    rest = {k: v for k, v in cf.items() if k not in printed}
    if rest:
        # don't spam too much; show a compact JSON line
        compact = json.dumps(rest, ensure_ascii=False)
        print(f"  other: {compact}")
    print()

    # fuzzy inputs
    if show_inputs_fuzzy:
        print("FUZZY INPUTS (μ)")
        for var_name, term_mu in result["fuzzy_inputs"].items():
            print(f"  {var_name}:")
            print(f"    {_format_term_line(term_mu, top_k=top_k_terms)}")
        print()

    # output aggregation (mandatory)
    print("OUTPUT AGGREGATION (μ)")
    for out_name, term_mu in result["output_aggregation"].items():
        print(f"  {out_name}:")
        print(f"    {_format_term_line(term_mu, top_k=top_k_terms)}")
    print()

    # crisp outputs
    print("DEFUZZIFIED OUTPUTS (crisp)")
    for k, v in result["outputs_crisp"].items():
        print(f"  {k:26s} {_fmt_float(float(v), 3)}")
    print()

    # top rules
    if top_k_rules and top_k_rules > 0:
        contrib = result.get("top_rules", [])
        if contrib:
            print("TOP RULES (contribution)")
            for item in contrib[:top_k_rules]:
                rid = item["id"]
                strength = item["strength"]
                then_map = item.get("then", {})
                then_s = ", ".join([f"{k}={v}" for k, v in then_map.items()])
                print(f"  [{_fmt_float(float(strength), 2)}] {rid}: {then_s}")
            print()

    print(line)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_stage3",
        description="Stage 3 fuzzy inference: features.jsonl -> final metrics (single review or batch).",
    )
    ap.add_argument(
        "--features",
        default=os.path.join("feature_create", "features.jsonl"),
        help="Path to features.jsonl",
    )
    ap.add_argument(
        "--rules",
        default=os.path.join(os.path.dirname(__file__), "rules.yml"),
        help="Path to rules.yml",
    )
    ap.add_argument(
        "--id",
        type=int,
        default=None,
        help="Review id to process (single mode). If omitted, runs batch mode.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output jsonl path for batch mode (default: fuzzy/final_metrics.jsonl).",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty terminal report (single mode).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON (single mode).",
    )
    ap.add_argument(
        "--top-k-terms",
        type=int,
        default=0,
        help="Show only top-K terms for each fuzzy set (0 = show all >= min threshold).",
    )
    ap.add_argument(
        "--top-k-rules",
        type=int,
        default=5,
        help="Show top-K fired rules in pretty report (0 = hide).",
    )
    ap.add_argument(
        "--min-mu",
        type=float,
        default=0.01,
        help="Minimum μ to display (pretty output).",
    )
    ap.add_argument(
        "--no-inputs-fuzzy",
        action="store_true",
        help="Do not print fuzzy inputs block in pretty report.",
    )
    ap.add_argument(
        "--no-derived",
        action="store_true",
        help="Do not print derived features in pretty report.",
    )
    ap.add_argument(
        "--show-text",
        action="store_true",
        help="Print 'text' field if present in the features JSON.",
    )
    args = ap.parse_args(argv)

    engine = FuzzyEngine.from_rules_file(args.rules)

    rows = read_jsonl(args.features)

    # single mode
    if args.id is not None:
        feat = find_by_id(rows, args.id)
        if feat is None:
            print(f"[ERROR] id={args.id} not found in {args.features}", file=sys.stderr)
            return 2

        result = engine.run_single(feat, top_k_rules=max(args.top_k_rules, 0))
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if args.pretty or not args.json:
            _pretty_report(
                engine=engine,
                features=feat,
                result=result,
                show_text=bool(args.show_text),
                show_inputs_fuzzy=not bool(args.no_inputs_fuzzy),
                show_derived=not bool(args.no_derived),
                top_k_terms=max(args.top_k_terms, 0),
                top_k_rules=max(args.top_k_rules, 0),
            )
        return 0

    # batch mode
    out_path = args.out or os.path.join("fuzzy", "final_metrics.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out_rows = []
    for feat in rows:
        out_rows.append(engine.run_single(feat, top_k_rules=0))

    write_jsonl(out_path, out_rows)
    print(f"[OK] Saved: {out_path} (jsonl)  n={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

