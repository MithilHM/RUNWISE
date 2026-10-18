"""
Seed Data Generator for RUNWISE.
Conforms to PRD Section 14 and Section 4 of Person A specification.

Generates:
1. Ground-truth demo scenario:
   - Experiment E1 (baseline_xgboost) uses dataset version D1 (dataset_v1).
   - Experiment E2 (tuned_xgboost) uses D1 and configuration derived from E1 (parent = E1).
   - Experiment E3 (distilled_xgboost) derived from E2 (parent = E2) and produces Claim C1.
   - Experiment E_control (resnet_vision) uses dataset version D2 (dataset_v2), completely unaffected.
2. A diverse batch of 50 synthetic experiments across multiple model families:
   - gradient_boosting, transformer, cnn, random_forest, linear_regression
   - multiple dataset versions
   - varying metrics, compute costs, and parent relationships
"""

import json
from typing import List, Dict, Any


def get_demo_lineage_scenario(project_id: str = "project_demo") -> List[Dict[str, Any]]:
    """
    Returns the core demonstrator scenario:
    - E1 uses D1
    - E2 uses D1 and derives from E1
    - E3 derives from E2 and produces a claim
    - E_control uses D2 and is independent
    """
    return [
        {
            "id": "exp_001",
            "project_id": project_id,
            "name": "baseline_xgboost",
            "model_family": "gradient_boosting",
            "dataset_name": "tabular_fraud",
            "dataset_version": "v1",
            "status": "completed",
            "duration_seconds": 120.0,
            "compute_cost_gpu_hours": 0.8,
            "random_seed": 42,
            "configuration": {
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8
            },
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.912,
                    "direction": "maximize",
                    "evaluation_split": "validation",
                    "evaluation_protocol": "stratified_5fold"
                },
                {
                    "name": "val_loss",
                    "value": 0.284,
                    "direction": "minimize",
                    "evaluation_split": "validation"
                }
            ],
            "description": "Baseline gradient boosted trees on tabular fraud dataset v1"
        },
        {
            "id": "exp_002",
            "project_id": project_id,
            "name": "tuned_xgboost_depth6",
            "model_family": "gradient_boosting",
            "dataset_name": "tabular_fraud",
            "dataset_version": "v1",
            "parent_experiment_id": "exp_001",
            "status": "completed",
            "duration_seconds": 210.0,
            "compute_cost_gpu_hours": 1.2,
            "random_seed": 42,
            "configuration": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8
            },
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.938,
                    "direction": "maximize",
                    "evaluation_split": "validation",
                    "evaluation_protocol": "stratified_5fold"
                },
                {
                    "name": "val_loss",
                    "value": 0.215,
                    "direction": "minimize",
                    "evaluation_split": "validation"
                }
            ],
            "description": "Tuned tree depth based on exp_001 baseline (single-variable change max_depth: 4 -> 6)"
        },
        {
            "id": "exp_003",
            "project_id": project_id,
            "name": "distilled_fast_tree",
            "model_family": "gradient_boosting",
            "dataset_name": "tabular_fraud",
            "dataset_version": "v1",
            "parent_experiment_id": "exp_002",
            "status": "completed",
            "duration_seconds": 95.0,
            "compute_cost_gpu_hours": 0.5,
            "random_seed": 42,
            "configuration": {
                "max_depth": 3,
                "distillation_source": "exp_002",
                "n_estimators": 80
            },
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.925,
                    "direction": "maximize",
                    "evaluation_split": "validation"
                }
            ],
            "description": "Compressed model distilled from exp_002 for latency optimization"
        },
        {
            "id": "exp_control",
            "project_id": project_id,
            "name": "vision_resnet_classifier",
            "model_family": "cnn",
            "dataset_name": "satellite_imagery",
            "dataset_version": "v2",
            "status": "completed",
            "duration_seconds": 1800.0,
            "compute_cost_gpu_hours": 2.5,
            "random_seed": 101,
            "configuration": {
                "backbone": "resnet50",
                "batch_size": 64,
                "optimizer": "adamw",
                "lr": 0.0003
            },
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.884,
                    "direction": "maximize",
                    "evaluation_split": "validation"
                }
            ],
            "description": "Control computer vision experiment completely isolated from tabular dataset"
        }
    ]


def generate_synthetic_experiments(project_id: str = "project_demo", total: int = 50) -> List[Dict[str, Any]]:
    """Generates 50 structured synthetic experiments across diverse model families."""
    experiments = get_demo_lineage_scenario(project_id)

    families = [
        ("transformer", ["bert_tiny", "distilbert", "roberta_base", "gpt2_small"], "nlp_sentiment"),
        ("cnn", ["convnext_tiny", "efficientnet_b0", "mobilenet_v3"], "image_cifar"),
        ("gradient_boosting", ["lightgbm_regressor", "catboost_classifier"], "tabular_sales"),
        ("random_forest", ["rf_deep", "rf_shallow", "extra_trees"], "genomics_microarray"),
        ("linear", ["ridge_regression", "elastic_net", "logistic_reg"], "housing_prices")
    ]

    count = len(experiments)
    fam_idx = 0

    while count < total:
        fam_name, arch_list, ds_name = families[fam_idx % len(families)]
        fam_idx += 1
        idx = count + 1
        v_tag = f"v{((idx % 3) + 1)}"
        arch = arch_list[idx % len(arch_list)]
        exp_id = f"exp_{idx:03d}"

        # Assign a parent for roughly half the runs
        parent_id = None
        if idx > 6 and idx % 2 == 0:
            parent_id = f"exp_{max(1, idx - 2):03d}"

        acc = round(0.70 + ((idx * 17) % 25) / 100.0, 3)
        loss = round(0.50 - ((idx * 13) % 30) / 100.0, 3)
        gpu_time = round(0.2 + (idx % 7) * 0.4, 2)

        experiments.append({
            "id": exp_id,
            "project_id": project_id,
            "name": f"{arch}_{v_tag}",
            "model_family": fam_name,
            "dataset_name": ds_name,
            "dataset_version": v_tag,
            "status": "completed" if idx % 10 != 0 else "failed",
            "duration_seconds": int(gpu_time * 3600),
            "compute_cost_gpu_hours": gpu_time,
            "random_seed": (idx * 42) % 1000,
            "parent_experiment_id": parent_id,
            "configuration": {
                "architecture": arch,
                "learning_rate": round(10 ** (- (idx % 4 + 2)), 5),
                "batch_size": 16 * (1 + (idx % 4)),
                "optimizer": "adam" if idx % 2 == 0 else "sgd"
            },
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": acc,
                    "direction": "maximize",
                    "evaluation_split": "validation"
                },
                {
                    "name": "validation_loss",
                    "value": loss,
                    "direction": "minimize",
                    "evaluation_split": "validation"
                }
            ],
            "description": f"Synthetic run {idx} evaluating {arch} on {ds_name}:{v_tag}"
        })
        count += 1

    return experiments
