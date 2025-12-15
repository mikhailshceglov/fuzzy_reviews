from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

EPS = 1e-12

# Магнитуда (pos+neg)/tokens_alpha может быть >1 для “плотных” отзывов.
# Нормируем в [0..1] через деление на MAGNITUDE_NORM_DIV и клиппинг.
MAGNITUDE_NORM_DIV = 0.35  # можно подкрутить под ваш датасет (0.25..0.6 обычно ок)


@dataclass(frozen=True)
class Triangle:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class Trapezoid:
    a: float
    b: float
    c: float
    d: float


def _tri_mu(x: float, t: Triangle) -> float:
    a, b, c = t.a, t.b, t.c
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a + EPS)
    return (c - x) / (c - b + EPS)


def _trap_mu(x: float, t: Trapezoid) -> float:
    a, b, c, d = t.a, t.b, t.c, t.d

    if x < a or x > d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if a <= x < b:
        denom = (b - a)
        return 1.0 if abs(denom) < EPS else (x - a) / (denom + EPS)

    denom = (d - c)
    return 1.0 if abs(denom) < EPS else (d - x) / (denom + EPS)


def mu(x: float, shape: Any) -> float:
    if isinstance(shape, Triangle):
        return max(0.0, min(1.0, _tri_mu(x, shape)))
    if isinstance(shape, Trapezoid):
        return max(0.0, min(1.0, _trap_mu(x, shape)))
    raise TypeError(f"Unknown shape: {type(shape)}")


# Входные лингвистические переменные (термы + функции принадлежности)
# Термы — только текстовые; численные параметры живут в коде.
INPUT_MFS: Dict[str, Dict[str, Any]] = {
    "score_crisp": {
        "сильно_негативная": Trapezoid(-1.0, -1.0, -0.85, -0.55),
        "негативная": Triangle(-0.85, -0.55, -0.15),
        "нейтральная": Triangle(-0.25, 0.0, 0.25),
        "позитивная": Triangle(0.15, 0.55, 0.85),
        "сильно_позитивная": Trapezoid(0.55, 0.85, 1.0, 1.0),
    },
    "coverage": {
        "низкое": Trapezoid(0.0, 0.0, 0.25, 0.45),
        "среднее": Triangle(0.25, 0.55, 0.85),
        "высокое": Trapezoid(0.65, 0.85, 1.0, 1.0),
    },
    "hedges_rate": {
        "мало": Trapezoid(0.0, 0.0, 0.03, 0.08),
        "средне": Triangle(0.04, 0.12, 0.22),
        "много": Trapezoid(0.16, 0.28, 1.0, 1.0),
    },
    "magnitude": {
        "низкая": Trapezoid(0.0, 0.0, 0.25, 0.45),
        "средняя": Triangle(0.25, 0.55, 0.85),
        "высокая": Trapezoid(0.65, 0.85, 1.0, 1.0),
    },
    "conflict": {
        "низкий": Trapezoid(0.0, 0.0, 0.25, 0.45),
        "средний": Triangle(0.25, 0.55, 0.85),
        "высокий": Trapezoid(0.65, 0.85, 1.0, 1.0),
    },
    # Обрати внимание: имя переменной — emotion_intensity_crisp (как в правилах),
    # а значение мы берём из features с fallback-логикой.
    #
    # КАЛИБРОВКА ПОД ВАШ ДАТАСЕТ (n=90013):
    #   q25 = 0.178880
    #   q50 = 0.233782
    #   q75 = 0.300396
    #   q90 = 0.364716
    #
    # Раньше стояли "универсальные" границы 0.25/0.45/0.65..., из-за чего почти
    # все значения попадали в "низкая" (μ≈1.0). Теперь термы подстроены под
    # реальные квантили.
    "emotion_intensity_crisp": {
        # левое плечо до ~медианы
        "низкая": Trapezoid(0.0, 0.0, 0.17888, 0.233782),
        # середина: пик около q75, затухание к q90
        "средняя": Triangle(0.17888, 0.300396, 0.364716),
        # правое плечо от q75..q90 и выше
        "высокая": Trapezoid(0.300396, 0.364716, 1.0, 1.0),
    },
}


def compute_derived_features(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Derived:
      hedges_rate  = hedges_count / (tokens_alpha + eps)
      magnitude    = clip01( ((pos+neg)/(tokens_alpha+eps)) / MAGNITUDE_NORM_DIV )
      conflict     = 1 - |pos-neg|/(pos+neg+eps)   (если pos+neg ~ 0 -> 0)
    """
    tokens = float(features.get("tokens_alpha", 0) or 0.0)
    hedges_count = float(features.get("hedges_count", 0) or 0.0)
    pos = float(features.get("pos", 0) or 0.0)
    neg = float(features.get("neg", 0) or 0.0)

    hedges_rate = hedges_count / (tokens + EPS)

    mag_raw = (pos + neg) / (tokens + EPS)  # “плотность” сентимента
    magnitude = mag_raw / (MAGNITUDE_NORM_DIV + EPS)
    if magnitude < 0:
        magnitude = 0.0
    if magnitude > 1:
        magnitude = 1.0

    s = pos + neg
    if s <= 1e-9:
        conflict = 0.0
    else:
        conflict = 1.0 - abs(pos - neg) / (s + EPS)
        conflict = max(0.0, min(1.0, conflict))

    return {
        "hedges_rate": float(hedges_rate),
        "magnitude": float(magnitude),
        "conflict": float(conflict),
    }


def _get_emotion_intensity_value(features: Dict[str, Any]) -> float:
    """
    Приоритет:
      1) emotion_intensity_crisp
      2) emotion_intensity_score
      3) intensity
      4) 0.0
    """
    for key in ("emotion_intensity_crisp", "emotion_intensity_score", "intensity"):
        if key in features and features[key] is not None:
            try:
                return float(features[key])
            except Exception:
                continue
    return 0.0


def extract_crisp_inputs(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Возвращает “входные числа” для fuzzy-ядра под именами переменных, используемых в rules.yml.
    """
    score_crisp = float(features.get("score_crisp", 0.0) or 0.0)
    coverage = float(features.get("coverage", 0.0) or 0.0)

    derived = compute_derived_features(features)
    emotion_intensity = _get_emotion_intensity_value(features)

    return {
        "score_crisp": score_crisp,
        "coverage": coverage,
        "hedges_rate": derived["hedges_rate"],
        "magnitude": derived["magnitude"],
        "conflict": derived["conflict"],
        "emotion_intensity_crisp": float(emotion_intensity),
    }


def fuzzify_variable(var_name: str, x: float) -> Dict[str, float]:
    mfs = INPUT_MFS.get(var_name)
    if mfs is None:
        raise KeyError(f"Unknown input variable: {var_name}")
    return {term: mu(x, shape) for term, shape in mfs.items()}


def fuzzify_review_inputs(
    features: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """
    Returns:
      fuzzy_inputs: var -> {term: μ}
      crisp_inputs: var -> crisp
      derived: derived features dict
    """
    derived = compute_derived_features(features)
    crisp_inputs = extract_crisp_inputs(features)

    fuzzy_inputs: Dict[str, Dict[str, float]] = {}
    for var_name, x in crisp_inputs.items():
        fuzzy_inputs[var_name] = fuzzify_variable(var_name, float(x))

    return fuzzy_inputs, crisp_inputs, derived
