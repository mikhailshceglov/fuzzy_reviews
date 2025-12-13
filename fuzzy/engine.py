from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .core import Triangle, Trapezoid, mu, fuzzify_review_inputs

try:
    import yaml  # type: ignore
except Exception as e:
    yaml = None  # noqa: N816


EPS = 1e-12


# Выходные функции принадлежности (для дефаззификации centroid)
OUTPUT_MFS: Dict[str, Dict[str, Any]] = {
    "final_tonality": {
        "сильно_негативная": Trapezoid(-1.0, -1.0, -0.85, -0.55),
        "негативная": Triangle(-0.85, -0.55, -0.15),
        "нейтральная": Triangle(-0.25, 0.0, 0.25),
        "позитивная": Triangle(0.15, 0.55, 0.85),
        "сильно_позитивная": Trapezoid(0.55, 0.85, 1.0, 1.0),
    },
    "trust": {
        "низкое": Trapezoid(0.0, 0.0, 0.25, 0.45),
        "среднее": Triangle(0.25, 0.55, 0.85),
        "высокое": Trapezoid(0.65, 0.85, 1.0, 1.0),
    },
    "final_emotion_intensity": {
        "низкая": Trapezoid(0.0, 0.0, 0.25, 0.45),
        "средняя": Triangle(0.25, 0.55, 0.85),
        "высокая": Trapezoid(0.65, 0.85, 1.0, 1.0),
    },
    "impression_strength": {
        "низкая": Trapezoid(0.0, 0.0, 0.25, 0.45),
        "средняя": Triangle(0.25, 0.55, 0.85),
        "высокая": Trapezoid(0.65, 0.85, 1.0, 1.0),
    },
}

OUTPUT_RANGES: Dict[str, Tuple[float, float]] = {
    "final_tonality": (-1.0, 1.0),
    "trust": (0.0, 1.0),
    "final_emotion_intensity": (0.0, 1.0),
    "impression_strength": (0.0, 1.0),
}


def _center_of_shape(shape: Any) -> float:
    # Для дефолтов/фолбэка: грубый “центр” терма
    if isinstance(shape, Triangle):
        return shape.b
    if isinstance(shape, Trapezoid):
        # центр плато (b..c)
        return 0.5 * (shape.b + shape.c)
    raise TypeError(type(shape))


@dataclass
class Stage3Result:
    review_id: Any
    crisp_features: Dict[str, Any]
    derived_features: Dict[str, float]
    fuzzy_inputs: Dict[str, Dict[str, float]]
    output_aggregation: Dict[str, Dict[str, float]]
    outputs_crisp: Dict[str, float]
    top_rules: List[Dict[str, Any]]


class FuzzyEngine:
    def __init__(self, rules_doc: Dict[str, Any]) -> None:
        self.rules_doc = rules_doc
        self.outputs_spec = rules_doc.get("outputs", {}) or {}
        self.rules = rules_doc.get("rules", []) or []
        self.defaults = rules_doc.get("defaults", {}) or {}

        # Validate minimal
        if not isinstance(self.rules, list) or not self.rules:
            raise ValueError("rules.yml: 'rules' must be a non-empty list")

    @staticmethod
    def from_rules_file(path: str) -> "FuzzyEngine":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not installed. Install it: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            raise ValueError("rules.yml: root must be a mapping")
        return FuzzyEngine(doc)

    def _eval_clause(self, fuzzy_inputs: Dict[str, Dict[str, float]], clause: Dict[str, Any]) -> float:
        # Supports:
        #   {var: X, is: TERM}
        #   {all: [ ... ]}
        #   {any: [ ... ]}
        #   {not: {...}}
        if "var" in clause and "is" in clause:
            var = str(clause["var"])
            term = str(clause["is"])
            return float(fuzzy_inputs.get(var, {}).get(term, 0.0))

        if "all" in clause:
            items = clause["all"] or []
            if not items:
                return 1.0
            vals = [self._eval_clause(fuzzy_inputs, c) for c in items]
            return float(min(vals)) if vals else 1.0

        if "any" in clause:
            items = clause["any"] or []
            if not items:
                return 0.0
            vals = [self._eval_clause(fuzzy_inputs, c) for c in items]
            return float(max(vals)) if vals else 0.0

        if "not" in clause:
            val = self._eval_clause(fuzzy_inputs, clause["not"])
            return float(1.0 - val)

        # Unknown clause
        return 0.0

    def _eval_if(self, fuzzy_inputs: Dict[str, Dict[str, float]], if_block: Dict[str, Any]) -> float:
        # if: {all:[...]} or {any:[...]} or direct clause
        if not isinstance(if_block, dict):
            return 0.0
        # Direct support:
        if "all" in if_block or "any" in if_block or "var" in if_block or "not" in if_block:
            return self._eval_clause(fuzzy_inputs, if_block)
        return 0.0

    def infer_outputs(
        self,
        fuzzy_inputs: Dict[str, Dict[str, float]],
        *,
        top_k_rules: int = 0,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        """
        Mamdani MAX–MIN inference:
          strength = eval(if)
          strength *= weight (optional), clipped to [0..1]
          then: output_var -> term
          output_term_mu = max over rules of min(strength, 1)
        """
        output_sets: Dict[str, Dict[str, float]] = {}
        fired_rules: List[Dict[str, Any]] = []

        for rule in self.rules:
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", ""))
            if_block = rule.get("if", {}) or {}
            then_block = rule.get("then", {}) or {}
            weight = float(rule.get("weight", 1.0) or 1.0)

            strength = self._eval_if(fuzzy_inputs, if_block)
            strength = max(0.0, min(1.0, strength * weight))

            if strength <= 0.0:
                continue

            # Apply conclusions
            if isinstance(then_block, dict):
                for out_var, out_term in then_block.items():
                    out_var = str(out_var)
                    out_term = str(out_term)

                    if out_var not in output_sets:
                        output_sets[out_var] = {}
                    prev = float(output_sets[out_var].get(out_term, 0.0))
                    output_sets[out_var][out_term] = max(prev, strength)

            fired_rules.append(
                {
                    "id": rid or "<no-id>",
                    "strength": strength,
                    "then": dict(then_block) if isinstance(then_block, dict) else {},
                }
            )

        fired_rules.sort(key=lambda x: float(x["strength"]), reverse=True)
        top_rules = fired_rules[:top_k_rules] if top_k_rules and top_k_rules > 0 else []
        return output_sets, top_rules

    def _defuzzify_centroid(
        self,
        out_var: str,
        term_mu: Dict[str, float],
        *,
        n: int = 801,
    ) -> float:
        """
        Centroid defuzzification over aggregated output membership function:
          μ_out(x) = max_t min( μ_term[t], mf_t(x) )
        """
        if out_var not in OUTPUT_MFS:
            raise KeyError(f"Unknown output var: {out_var}")

        lo, hi = OUTPUT_RANGES[out_var]
        mfs = OUTPUT_MFS[out_var]

        # If nothing activated: fallback to default term center if exists, else middle of range.
        if not term_mu or max(term_mu.values(), default=0.0) <= 0.0:
            dterm = self.defaults.get(out_var)
            if dterm and dterm in mfs:
                return float(_center_of_shape(mfs[dterm]))
            return float(0.5 * (lo + hi))

        xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        num = 0.0
        den = 0.0

        for x in xs:
            agg = 0.0
            for term, alpha in term_mu.items():
                if alpha <= 0.0:
                    continue
                shape = mfs.get(term)
                if shape is None:
                    continue
                agg = max(agg, min(float(alpha), mu(x, shape)))
            num += x * agg
            den += agg

        if den <= EPS:
            # fallback to the strongest term center
            best_term = max(term_mu.items(), key=lambda kv: kv[1])[0]
            if best_term in mfs:
                return float(_center_of_shape(mfs[best_term]))
            return float(0.5 * (lo + hi))

        return float(num / den)

    def run_single(self, features: Dict[str, Any], *, top_k_rules: int = 5) -> Dict[str, Any]:
        """
        Full stage 3 pipeline for one review features dict.
        Returns a JSON-serializable dict:
          - crisp_features
          - derived_features
          - fuzzy_inputs
          - output_aggregation
          - outputs_crisp
          - top_rules (optional)
        """
        fuzzy_inputs, crisp_inputs, derived = fuzzify_review_inputs(features)

        output_sets, top_rules = self.infer_outputs(fuzzy_inputs, top_k_rules=top_k_rules)

        # Ensure every expected output exists (fill empty -> {})
        expected_outputs = ["final_tonality", "trust", "final_emotion_intensity", "impression_strength"]
        for out in expected_outputs:
            output_sets.setdefault(out, {})

            # If still empty, seed with defaults at μ=0 (so printing is stable)
            if not output_sets[out] and out in self.defaults:
                output_sets[out][str(self.defaults[out])] = 0.0

        outputs_crisp = {
            "final_tonality_crisp": self._defuzzify_centroid("final_tonality", output_sets["final_tonality"]),
            "trust": self._defuzzify_centroid("trust", output_sets["trust"]),
            "final_emotion_intensity": self._defuzzify_centroid("final_emotion_intensity", output_sets["final_emotion_intensity"]),
            "impression_strength": self._defuzzify_centroid("impression_strength", output_sets["impression_strength"]),
        }

        # We keep original crisp features for transparency (not only crisp_inputs)
        crisp_features = dict(features)

        return {
            "id": features.get("id"),
            "crisp_features": crisp_features,
            "derived_features": derived,
            "fuzzy_inputs": fuzzy_inputs,
            "output_aggregation": output_sets,
            "outputs_crisp": outputs_crisp,
            "top_rules": top_rules,
        }
