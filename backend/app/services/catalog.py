"""The fixed catalog: three models and their hyperparameter spaces.

This module is the authority for three things at once:

1. what `GET /api/v1/catalog/models` returns,
2. what the frontend renders controls from,
3. what the graph stores as `Param` nodes.

Keeping one declarative definition means a parameter cannot exist in the UI but
not in the search space, or in the search space but not in the graph.

Adding a fourth model means editing `MODEL_SPECS` and nothing else. The
hyperparameters of a model are its own; there is no shared gradient-boosting
parameter block, because a shared block is how `max_depth` silently ends up
meaning something different for two libraries.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from app.models.contracts import (
    BINARY_ONLY_METRICS,
    MetricName,
    ModelSpec,
    ParamSpec,
)

# The only two classification shapes this harness handles.
TASKS = ("binary", "multiclass")

__all__ = [
    "MODEL_SPECS",
    "MODEL_IDS",
    "METRICS",
    "METRIC_HELP",
    "TASKS",
    "get_model",
    "list_models",
    "default_params",
    "coerce_params",
    "validate_params",
    "build_estimator",
    "available_metrics",
    "sample_params",
    "repair_params",
]


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------

_RANDOM_FOREST = ModelSpec(
    id="random_forest",
    name="RandomForest",
    library="sklearn.ensemble",
    description=(
        "Bagged decision trees on bootstrap samples with randomised feature splits. "
        "The baseline: almost nothing to tune, very hard to get catastrophically wrong, "
        "and its errors are readable."
    ),
    params=[
        ParamSpec(name="n_estimators", kind="int", low=20, high=500, step=20, default=200),
        ParamSpec(name="max_depth", kind="int_or_null", low=2, high=32, step=1, default=None),
        ParamSpec(name="min_samples_split", kind="int", low=2, high=40, step=1, default=2),
        ParamSpec(name="min_samples_leaf", kind="int", low=1, high=20, step=1, default=1),
        ParamSpec(
            name="max_features",
            kind="categorical",
            choices=["sqrt", "log2", 0.3, 0.5, 0.8, 1.0],
            default="sqrt",
        ),
        ParamSpec(name="criterion", kind="categorical", choices=["gini", "entropy", "log_loss"], default="gini"),
        ParamSpec(
            name="class_weight",
            kind="categorical",
            choices=[None, "balanced", "balanced_subsample"],
            default=None,
        ),
        ParamSpec(name="bootstrap", kind="bool", default=True),
    ],
)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

_XGBOOST = ModelSpec(
    id="xgboost",
    name="XGBoost",
    library="xgboost",
    description=(
        "Gradient-boosted trees with second-order split gains and explicit L1/L2 "
        "regularisation. The usual first choice on tabular data; sensitive to depth "
        "and learning rate, so the most to tune here."
    ),
    params=[
        ParamSpec(name="n_estimators", kind="int", low=50, high=1000, step=50, default=300),
        ParamSpec(name="max_depth", kind="int", low=2, high=12, step=1, default=6),
        ParamSpec(name="learning_rate", kind="float", low=0.01, high=0.5, log_scale=True, default=0.1),
        ParamSpec(name="min_child_weight", kind="float", low=0.5, high=20.0, step=0.5, default=1.0),
        ParamSpec(name="subsample", kind="float", low=0.4, high=1.0, step=0.05, default=1.0),
        ParamSpec(name="colsample_bytree", kind="float", low=0.3, high=1.0, step=0.05, default=1.0),
        ParamSpec(name="reg_alpha", kind="float", low=0.0, high=10.0, step=0.1, default=0.0),
        ParamSpec(name="reg_lambda", kind="float", low=0.1, high=50.0, log_scale=True, default=1.0),
        ParamSpec(name="gamma", kind="float", low=0.0, high=5.0, step=0.1, default=0.0),
    ],
)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

_LIGHTGBM = ModelSpec(
    id="lightgbm",
    name="LightGBM",
    library="lightgbm",
    description=(
        "Histogram-based boosting that grows leaves rather than levels, with "
        "leaf-wise growth. Usually finds the same optimum as XGBoost with fewer "
        "trees and a different failure mode, which is why it is worth running both."
    ),
    params=[
        ParamSpec(name="n_estimators", kind="int", low=50, high=1000, step=50, default=300),
        ParamSpec(name="num_leaves", kind="int", low=7, high=255, step=1, default=31),
        ParamSpec(name="learning_rate", kind="float", low=0.01, high=0.5, log_scale=True, default=0.1),
        ParamSpec(name="max_depth", kind="int", low=2, high=16, step=1, default=8),
        ParamSpec(name="min_child_samples", kind="int", low=5, high=100, step=1, default=20),
        ParamSpec(name="subsample", kind="float", low=0.4, high=1.0, step=0.05, default=0.9),
        ParamSpec(name="colsample_bytree", kind="float", low=0.3, high=1.0, step=0.05, default=1.0),
        ParamSpec(name="reg_alpha", kind="float", low=0.0, high=10.0, step=0.1, default=0.0),
        ParamSpec(name="reg_lambda", kind="float", low=0.01, high=50.0, log_scale=True, default=1.0),
    ],
)


MODEL_SPECS: List[ModelSpec] = [_RANDOM_FOREST, _XGBOOST, _LIGHTGBM]
MODEL_IDS: List[str] = [spec.id for spec in MODEL_SPECS]

_BY_ID: Dict[str, ModelSpec] = {spec.id: spec for spec in MODEL_SPECS}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS: List[MetricName] = ["accuracy", "f1_macro", "roc_auc", "pr_auc"]

METRIC_HELP: Dict[str, str] = {
    "accuracy": "Share of correct predictions. Misleading on an imbalanced target.",
    "f1_macro": "Unweighted mean F1 across classes. Penalises ignoring rare classes.",
    "roc_auc": "Ranking quality of the positive score. One-vs-rest weighted on multiclass.",
    "pr_auc": "Area under the precision-recall curve. Binary targets only.",
}

# A classification target with more distinct values than this is not a class
# label. Ingest rejects it rather than training a model against a continuous
# column and reporting a meaningless accuracy against 200 classes.
MAX_CLASSES = 50
MIN_CLASS_MEMBERS = 2


def available_metrics(task: str) -> List[MetricName]:
    """Metrics usable for a task.

    `pr_auc` is binary-only. It is filtered out here rather than substituted, so
    a multiclass run cannot silently report something that is not PR-AUC. An
    unknown task raises instead of quietly returning the multiclass list, because
    the caller is holding a typo and would otherwise get a confident-looking
    answer that hides it.
    """
    if task == "binary":
        return list(METRICS)
    if task == "multiclass":
        return [m for m in METRICS if m not in BINARY_ONLY_METRICS]
    raise ValueError(
        f"unknown task '{task}'; this harness handles "
        f"{', '.join(sorted(TASKS))} only"
    )


def sample_params(model_id: str, rng: random.Random) -> Dict[str, Any]:
    """Draws a configuration from the model's own space.

    Log-scaled parameters are sampled log-uniformly rather than uniformly. Over a
    range like `reg_lambda` in [0.1, 50] a uniform draw puts more than 98% of its
    mass below 1.0, so the search would never reach the regularisation values that
    actually matter. `log_scale` is honoured here for that reason.
    """
    spec = get_model(model_id)
    resolved: Dict[str, Any] = {}
    for param in spec.params:
        if param.kind == "bool":
            resolved[param.name] = rng.random() < 0.5
        elif param.kind == "categorical" and param.choices:
            resolved[param.name] = rng.choice(param.choices)
        elif param.kind in ("int", "float", "int_or_null") and param.low is not None and param.high is not None:
            low, high = float(param.low), float(param.high)
            if param.log_scale and low > 0 and high > 0:
                draw = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                draw = rng.uniform(low, high)
            if param.step:
                draw = float(param.low) + round((draw - float(param.low)) / float(param.step)) * float(param.step)
            resolved[param.name] = int(round(draw)) if param.kind in ("int", "int_or_null") else round(draw, 6)
        else:
            resolved[param.name] = param.default
    return coerce_params(model_id, resolved)


def repair_params(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves cross-parameter conflicts that `coerce_params` cannot see.

    LightGBM's `num_leaves` is capped by `max_depth`: a level-wise tree of depth
    d cannot have more than 2^d - 1 leaves. Drawing the two independently
    therefore produces an unusable pair a quarter of the time, which burns a
    budget step and writes a confident rationale for a configuration the harness
    itself rejects. Rather than report it, the leaf count is brought back under
    the ceiling.
    """
    resolved = coerce_params(model_id, params)
    if model_id == "lightgbm":
        num_leaves = resolved.get("num_leaves")
        max_depth = resolved.get("max_depth")
        if num_leaves is not None and max_depth is not None:
            ceiling = 2 ** int(max_depth) - 1
            if num_leaves > ceiling:
                resolved["num_leaves"] = max(int(2 ** int(max_depth) // 2), 2)
    return resolved


def get_model(model_id: str) -> ModelSpec:
    """Looks up a model, raising a readable error for an unknown id."""
    spec = _BY_ID.get(model_id)
    if spec is None:
        raise KeyError(
            f"unknown model '{model_id}'; the catalog has {', '.join(MODEL_IDS)}"
        )
    return spec


def list_models() -> List[ModelSpec]:
    return list(MODEL_SPECS)


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def default_params(model_id: str) -> Dict[str, Any]:
    """The values used when a researcher has not touched a control."""
    return {spec.name: spec.default for spec in get_model(model_id).params}


def coerce_params(model_id: str, raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Clamps a partial parameter dict onto the model's domain.

    Missing keys fall back to the default, out-of-range values are clamped, and
    keys the model does not have are dropped. This is the only path from the UI
    to the trainer, so a config that reaches `build_estimator` is always in domain.
    """
    spec = get_model(model_id)
    source: Dict[str, Any] = dict(raw or {})
    resolved: Dict[str, Any] = {}
    for param in spec.params:
        resolved[param.name] = param.clamp(source.get(param.name, param.default))
    return resolved


def validate_params(
    model_id: str, params: Dict[str, Any], strict: bool = False
) -> List[str]:
    """Returns human-readable problems with a configuration. Empty means usable.

    Two modes, and the difference matters:

    - **lenient** (default): also checks the cross-parameter rules that a
      per-parameter range cannot express. Use on an already-clamped dict.
    - **strict**: additionally rejects any value outside its declared domain. Use
      on a dict that arrived from a client rather than from our own UI.

    The reason for both: clamping is correct when a researcher drags a slider past
    its end, and wrong when a caller sends `learning_rate=0`. Clamping the second
    case silently runs a configuration nobody asked for.
    """
    spec = get_model(model_id)
    problems: List[str] = []

    unknown = [key for key in params if spec.param(key) is None]
    if unknown:
        problems.append(f"{spec.name} has no parameter(s): {', '.join(sorted(unknown))}")

    for key, value in params.items():
        param = spec.param(key)
        if param is None:
            continue
        if value is None:
            # `class_weight=None` is a real, meaningful setting, so a null is
            # only a problem when the parameter cannot take one.
            nullable = param.kind == "int_or_null" or (
                param.choices is not None and None in param.choices
            )
            if not nullable:
                problems.append(f"{key} may not be null")
            continue

        # Strict mode asks the spec, not a chain of comparisons. Comparing a
        # string against a float raises rather than returning False, so a
        # hand-written range check here would turn a bad request into a 500
        # instead of the 422 it should be.
        if strict:
            if not param.in_domain(value):
                problems.append(f"{key}={value!r} is outside its allowed domain")
                continue

        if param.kind in ("int", "float", "int_or_null"):
            if param.low is not None and value < param.low:
                problems.append(f"{key}={value} is below the minimum {param.low}")
            if param.high is not None and value > param.high:
                problems.append(f"{key}={value} is above the maximum {param.high}")
        elif param.kind == "categorical" and param.choices is not None:
            if value not in param.choices:
                problems.append(f"{key}={value!r} is not one of {param.choices}")
        elif param.kind == "bool" and not isinstance(value, bool):
            problems.append(f"{key} must be true or false, got {value!r}")

    # Cross-parameter rules.
    if model_id == "lightgbm":
        num_leaves = params.get("num_leaves")
        max_depth = params.get("max_depth")
        if num_leaves is not None and max_depth is not None:
            # Leaf-wise growth cannot produce more leaves than the level-wise tree
            # of the same depth allows, so the larger setting is simply ignored.
            # That is a silently dead knob, which is worse than a reported one.
            ceiling = 2**int(max_depth) - 1
            if num_leaves > ceiling:
                problems.append(
                    f"num_leaves={num_leaves} exceeds the {ceiling} reachable at "
                    f"max_depth={max_depth}, so it would be silently ignored"
                )
    if model_id == "xgboost":
        min_child_weight = params.get("min_child_weight")
        if min_child_weight is not None and float(min_child_weight) <= 0:
            problems.append("min_child_weight must be positive")

    return problems


def build_estimator(model_id: str, params: Dict[str, Any], seed: int = 42) -> Any:
    """Instantiates the estimator. Imports are local so a missing optional
    library surfaces as a clear error at trial time, not at import time."""
    spec = get_model(model_id)
    resolved = coerce_params(model_id, params)
    problems = validate_params(model_id, resolved)
    if problems:
        raise ValueError("; ".join(problems))

    if model_id == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=resolved["n_estimators"],
            max_depth=resolved["max_depth"],
            min_samples_split=resolved["min_samples_split"],
            min_samples_leaf=resolved["min_samples_leaf"],
            max_features=resolved["max_features"],
            criterion=resolved["criterion"],
            class_weight=resolved["class_weight"],
            bootstrap=resolved["bootstrap"],
            random_state=seed,
            n_jobs=2,
        )

    if model_id == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("xgboost is not installed") from exc

        return XGBClassifier(
            n_estimators=resolved["n_estimators"],
            max_depth=resolved["max_depth"],
            learning_rate=resolved["learning_rate"],
            min_child_weight=resolved["min_child_weight"],
            subsample=resolved["subsample"],
            colsample_bytree=resolved["colsample_bytree"],
            reg_alpha=resolved["reg_alpha"],
            reg_lambda=resolved["reg_lambda"],
            gamma=resolved["gamma"],
            random_state=seed,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
            n_jobs=2,
        )

    if model_id == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("lightgbm is not installed") from exc

        return LGBMClassifier(
            n_estimators=resolved["n_estimators"],
            num_leaves=resolved["num_leaves"],
            learning_rate=resolved["learning_rate"],
            max_depth=resolved["max_depth"],
            min_child_samples=resolved["min_child_samples"],
            subsample=resolved["subsample"],
            subsample_freq=1 if resolved["subsample"] < 1.0 else 0,
            colsample_bytree=resolved["colsample_bytree"],
            reg_alpha=resolved["reg_alpha"],
            reg_lambda=resolved["reg_lambda"],
            random_state=seed,
            verbosity=-1,
            n_jobs=2,
        )

    raise KeyError(f"no estimator for model '{model_id}' (have {spec.id})")