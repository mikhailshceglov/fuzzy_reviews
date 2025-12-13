from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_HAS_RAZDEL = False
_HAS_PYMORPHY = False

try:
    import pymorphy3 as pymorphy  # type: ignore
    _HAS_PYMORPHY = True
except Exception:
    try:
        import pymorphy2 as pymorphy  # type: ignore
        _HAS_PYMORPHY = True
    except Exception:
        _HAS_PYMORPHY = False

try:
    from razdel import tokenize as razdel_tokenize  # type: ignore
    _HAS_RAZDEL = True
except Exception:
    _HAS_RAZDEL = False

WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def normalize_text(s: str) -> str:
    return s.strip().lower()


def simple_tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if _HAS_RAZDEL:
        return [
            t.text.lower()
            for t in razdel_tokenize(text)
            if WORD_RE.fullmatch(t.text.lower() or "")
        ]  # type: ignore
    return simple_tokenize(text)


class Lemmatizer:
    def __init__(self) -> None:
        self.morph = pymorphy.MorphAnalyzer() if _HAS_PYMORPHY else None  # type: ignore

    def lemma(self, token: str) -> str:
        if self.morph is None:
            return token
        p = self.morph.parse(token)
        if not p:
            return token
        return p[0].normal_form


def load_modifiers_yaml_lite(path: Path) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    current: Optional[str] = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and not line.startswith("-"):
            key = line.split(":", 1)[0].strip()
            if key in {"negation", "intensifiers", "downtoners", "contrast_markers"}:
                current = key
                groups.setdefault(current, [])
            else:
                current = None
            continue

        if line.startswith("-") and current is not None:
            item = line[1:].strip().strip("'\"")
            if item:
                groups[current].append(item)

    for k in list(groups.keys()):
        groups[k] = [normalize_text(x) for x in groups[k]]

    return groups


def modifiers_as_sets(mods: Dict[str, List[str]]) -> Dict[str, set]:
    return {k: set(v) for k, v in mods.items()}


def load_lexicon_json(path: Path) -> Tuple[Dict[str, float], Dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pol: Dict[str, float] = {}
    freq: Dict[str, int] = {}

    def put(lemma: str, p: Any, f: Any = None) -> None:
        if lemma is None:
            return
        lemma_n = normalize_text(str(lemma))
        try:
            p_f = float(p)
        except Exception:
            return
        p_f = max(-1.0, min(1.0, p_f))
        pol[lemma_n] = p_f
        if f is not None:
            try:
                freq[lemma_n] = int(f)
            except Exception:
                pass

    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                if isinstance(it, dict):
                    put(it.get("lemma"), it.get("polarity"), it.get("freq"))
        else:
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    put(k, v, None)
                elif isinstance(v, dict):
                    put(k, v.get("polarity"), v.get("freq"))
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                put(it.get("lemma"), it.get("polarity"), it.get("freq"))

    return pol, freq


def load_lexicon_csv(path: Path) -> Tuple[Dict[str, float], Dict[str, int]]:
    pol: Dict[str, float] = {}
    freq: Dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return pol, freq

        fn = [x.strip().lower() for x in reader.fieldnames]

        def pick(*cands: str) -> Optional[str]:
            for c in cands:
                if c in fn:
                    return reader.fieldnames[fn.index(c)]
            return None

        col_lemma = pick("lemma", "лемма", "word", "token")
        col_pol = pick("polarity", "sentiment", "score", "pol")
        col_freq = pick("freq", "count", "frequency", "n")

        if col_lemma is None or col_pol is None:
            return pol, freq

        for row in reader:
            lemma = normalize_text(row.get(col_lemma, "") or "")
            if not lemma:
                continue
            try:
                p = float(row.get(col_pol, ""))
            except Exception:
                continue
            p = max(-1.0, min(1.0, p))
            pol[lemma] = p
            if col_freq is not None:
                try:
                    freq[lemma] = int(float(row.get(col_freq, "0") or 0))
                except Exception:
                    pass

    return pol, freq


@dataclass
class ReviewFeatures:
    id: int
    text: str
    pos: float
    neg: float
    score_crisp: float
    intensity: float
    emotion_intensity_crisp: float
    emotion_intensity_score: float
    tokens_alpha: int
    lex_hits: int
    coverage: float
    intensifiers_count: int
    downtoners_count: int
    hedges_count: int
    negations_count: int
    contrast_count: int


def _detect_phrases(tokens: List[str], phrase_set: set, max_n: int = 3) -> List[Tuple[int, int, str]]:
    hits: List[Tuple[int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        for k in range(min(max_n, n - i), 1, -1):
            joined = "_".join(tokens[i:i + k])
            if joined in phrase_set:
                hits.append((i, i + k, joined))
                i += k
                matched = True
                break
        if not matched:
            i += 1
    return hits


def extract_features_for_review(
    text: str,
    rid: int,
    lex_polarity: Dict[str, float],
    mods: Dict[str, set],
    lemmatizer: Lemmatizer,
    window: int = 3,
) -> ReviewFeatures:
    tokens = tokenize(text)
    lemmas = [lemmatizer.lemma(t) for t in tokens]

    all_phrase_mods = set().union(
        mods.get("intensifiers", set()),
        mods.get("downtoners", set()),
        mods.get("contrast_markers", set()),
        mods.get("negation", set()),
    )

    phrase_hits = _detect_phrases(tokens, all_phrase_mods, max_n=3)
    phrase_at: Dict[int, Tuple[int, str]] = {}
    for s, e, ph in phrase_hits:
        phrase_at[s] = (e, ph)

    collapsed_tokens: List[str] = []
    collapsed_lemmas: List[str] = []
    i = 0
    while i < len(tokens):
        if i in phrase_at:
            e, ph = phrase_at[i]
            collapsed_tokens.append(ph)
            collapsed_lemmas.append(ph)
            i = e
        else:
            collapsed_tokens.append(tokens[i])
            collapsed_lemmas.append(lemmas[i])
            i += 1

    negation = mods.get("negation", set())
    intensifiers = mods.get("intensifiers", set())
    downtoners = mods.get("downtoners", set())
    contrast = mods.get("contrast_markers", set())

    intensifiers_count = sum(1 for t in collapsed_tokens if t in intensifiers)
    downtoners_count = sum(1 for t in collapsed_tokens if t in downtoners)
    hedges_count = downtoners_count
    negations_count = sum(1 for t in collapsed_tokens if t in negation)
    contrast_count = sum(1 for t in collapsed_tokens if t in contrast)

    pos = 0.0
    neg = 0.0
    abs_sum = 0.0
    hits = 0

    tokens_alpha = sum(1 for t in collapsed_tokens if WORD_RE.fullmatch(t.replace("_", "")) is not None)

    for idx, lemma in enumerate(collapsed_lemmas):
        base = lex_polarity.get(lemma)
        if base is None:
            continue

        left = collapsed_tokens[max(0, idx - window): idx]

        is_negated = any(t in negation for t in left)
        has_intens = any(t in intensifiers for t in left)
        has_down = any(t in downtoners for t in left)

        p = float(base)

        if is_negated:
            p = -p
            p *= 0.9

        if has_intens and has_down:
            p *= 1.0
        elif has_intens:
            p *= 1.5
        elif has_down:
            p *= 0.75

        p = max(-1.0, min(1.0, p))

        hits += 1
        abs_sum += abs(p)
        if p >= 0:
            pos += p
        else:
            neg += -p

    denom = pos + neg
    score_crisp = (pos - neg) / (denom + 1e-12)
    intensity = (abs_sum / hits) if hits > 0 else 0.0
    coverage = (hits / tokens_alpha) if tokens_alpha > 0 else 0.0

    emotion_intensity_crisp = abs_sum / (hits + 1.0)
    emotion_intensity_crisp = min(1.0, emotion_intensity_crisp * (1.0 + 0.1 * intensifiers_count))
    emotion_intensity_score = emotion_intensity_crisp

    return ReviewFeatures(
        id=rid,
        text=text,
        pos=round(pos, 6),
        neg=round(neg, 6),
        score_crisp=round(score_crisp, 6),
        intensity=round(intensity, 6),
        emotion_intensity_crisp=round(emotion_intensity_crisp, 6),
        emotion_intensity_score=round(emotion_intensity_score, 6),
        tokens_alpha=int(tokens_alpha),
        lex_hits=int(hits),
        coverage=round(coverage, 6),
        intensifiers_count=int(intensifiers_count),
        downtoners_count=int(downtoners_count),
        hedges_count=int(hedges_count),
        negations_count=int(negations_count),
        contrast_count=int(contrast_count),
    )


def iter_reviews_lines(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            yield s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews-txt", type=Path, required=True)
    ap.add_argument("--lexicon-json", type=Path, default=None)
    ap.add_argument("--lexicon-csv", type=Path, default=None)
    ap.add_argument("--modifiers-yaml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()

    if args.lexicon_json is None and args.lexicon_csv is None:
        raise SystemExit("Нужно указать --lexicon-json или --lexicon-csv")

    lex_polarity: Dict[str, float] = {}
    lex_freq: Dict[str, int] = {}

    if args.lexicon_json is not None and args.lexicon_json.exists():
        p, f = load_lexicon_json(args.lexicon_json)
        lex_polarity.update(p)
        lex_freq.update(f)

    if not lex_polarity and args.lexicon_csv is not None and args.lexicon_csv.exists():
        p, f = load_lexicon_csv(args.lexicon_csv)
        lex_polarity.update(p)
        lex_freq.update(f)

    if not lex_polarity:
        raise SystemExit("Лексикон пустой или не распознан. Проверь JSON/CSV формат.")

    mods_raw = load_modifiers_yaml_lite(args.modifiers_yaml)
    mods = modifiers_as_sets(mods_raw)

    lemmatizer = Lemmatizer()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_example: Optional[ReviewFeatures] = None

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, review in enumerate(iter_reviews_lines(args.reviews_txt)):
            feats = extract_features_for_review(
                text=review,
                rid=i,
                lex_polarity=lex_polarity,
                mods=mods,
                lemmatizer=lemmatizer,
                window=args.window,
            )
            out_f.write(json.dumps(asdict(feats), ensure_ascii=False) + "\n")
            if first_example is None:
                first_example = feats

    print(f"[OK] Saved: {out_path} (jsonl)")
    print(f"[INFO] lexicon size: {len(lex_polarity)}; modifiers groups: {list(mods.keys())}")

    if first_example is not None:
        snippet = first_example.text
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        print("[EXAMPLE] review:", snippet)
        print("[EXAMPLE] features:", json.dumps(asdict(first_example), ensure_ascii=False))

    if not _HAS_PYMORPHY:
        print("[WARN] pymorphy (pymorphy3/pymorphy2) not found -> lemmatization fallback = token itself.")


if __name__ == "__main__":
    main()
