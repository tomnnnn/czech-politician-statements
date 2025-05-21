"""
Evaluation utils for the fact checker.

@author: Hai Phong Nguyen
"""

import json
import os
import random
from typing import Any, Literal

import dspy
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from dataset_manager import Dataset
from dataset_manager.models import Statement
import logging

# Disable noisy LiteLLM logs
loggers = [
    "LiteLLM Proxy",
    "LiteLLM Router",
    "LiteLLM",
    "httpx"
]

for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL + 1)

def get_allowed_labels(mode: Literal["binary", "multi"]) -> list[str]:
    """
    Get the allowed labels based on the mode.
    """
    if mode == "binary":
        return ["pravda", "nepravda"]
    elif mode == "multi":
        return ["pravda", "nepravda", "neověřielné"]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def setup_evaluation(
    model: str = "hosted_vllm/Qwen/Qwen2.5-32B-Instruct-AWQ",
    api_base: str = "http://localhost:8000/v1",
    experiment_name: str = "thesis",
    dataset_path: str = "datasets/dataset_demagog.sqlite",
    sample_types: list[Literal["balanced", "filtered", "random"]] = ["random"],
    sample_portions: list[float] = [0.2],
    min_num_evidence: int = 1,
    mode: Literal["binary", "multi"] = "binary",
    output_folder: str = "results/dspy",
) -> list[list[dspy.Example]]:
    """
    Setup the evaluation environment and create examples for the fact checker.

    Args:
        model (str): The model to use for evaluation.
        api_base (str): The API base URL for the model.
        experiment_name (str): The name of the experiment.
        dataset_path (str): The path to the dataset.
        sample_types (list[str]): The types of samples to create.
        sample_portions (list[float]): The portions of samples to create.
        min_num_evidence (int): The minimum number of evidence required.
        mode (str): The mode for evaluation ("binary" or "multi").
        output_folder (str): The folder to save the results.

    Returns:
        examples (list[dspy.Example]): The list of examples for evaluation.
    """

    samples_list = []

    for sample_type, sample_portion in zip(sample_types, sample_portions):
        statements = sample_statements(
            sample_type,
            portion=sample_portion,
            allowed_labels=get_allowed_labels(mode),
            dataset_path=dataset_path,
            min_num_evidence=min_num_evidence,
        )
        examples = create_examples(statements)
        samples_list.append(examples)

    # prepare output folder
    os.makedirs(output_folder, exist_ok=True)

    setup_mlflow(experiment_name=experiment_name)
    setup_dspy(api_base=api_base, model=model)

    return samples_list


def setup_dspy(
    model: str = "hosted_vllm/Qwen/Qwen2.5-32B-Instruct-AWQ",
    api_base: str = "http://localhost:8000/v1",
):
    model = os.environ.get("LLM_MODEL", model)
    api_base = os.environ.get("LLM_API_BASE", api_base)

    lm = dspy.LM(
        model,
        api_base=api_base,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0,
        max_tokens=3000,
    )
    dspy.configure(lm=lm)


def setup_mlflow(experiment_name: str = "thesis"):
    mlflow.set_tracking_uri("sqlite:///thesis_runs_test.db")
    mlflow.set_experiment(experiment_name)
    mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)


def sample_statements(
    sample_type: Literal["balanced", "filtered", "random"],
    dataset_path: str = "datasets/dataset_demagog.sqlite",
    portion: float = 0.2,
    allowed_labels: list[str] = ["pravda", "nepravda"],
    min_num_evidence=1,
) -> list[Statement]:
    # TODO: Remove hardcoded paths
    random.seed(42)
    dataset = Dataset(dataset_path)
    statements = dataset.get_statements(
        allowed_labels=allowed_labels, min_evidence_count=min_num_evidence
    )

    if sample_type == "balanced":
        label_map = {
            label: [s for s in statements if s.label.lower() == label]
            for label in allowed_labels
        }
        num_samples = int(len(statements) * portion)
        num_samples_per_label = num_samples // len(label_map)

        samples = []

        for subset in label_map.values():
            statements = shuffle(statements, random_state=42)
            samples += subset[:num_samples_per_label]

    elif sample_type == "filtered":
        with open("datasets/auto-curated/auto-curated.json", "r") as f:
            verifiable_statements = json.load(f)
            verifiable_ids = [item["id"] for item in verifiable_statements]

        statements = [s for s in statements if s.id in verifiable_ids]
        _, samples = (
            train_test_split(
                statements,
                test_size=portion,
                stratify=[s.label for s in statements],
                random_state=42,
            )
            if portion < 1
            else (None, statements)
        )

    elif sample_type == "random":
        _, samples = (
            train_test_split(
                statements,
                test_size=portion,
                stratify=[s.label for s in statements],
                random_state=42,
            )
            if portion < 1
            else (None, statements)
        )

    return samples


def create_examples(statements: list[Statement]):
    examples = [
        dspy.Example(
            statement=statement,
            label=statement.label,
        ).with_inputs("statement")
        for statement in statements
    ]
    return examples


def evaluate(
    fact_checker: dspy.Module,
    examples: list[dspy.Example],
    output_folder: str,
    mode: Literal["binary", "multi"] = "binary",
    name: str = "evaluation",
) -> tuple[Any, dict]:
    """
    Evaluate the fact checker on the provided examples and saves the results.
    """

    pred_labels = []
    ref_labels = []

    def metric(example, pred, trace=None):
        pred_labels.append(pred.label.lower())
        ref_labels.append(example.label.lower())

        return pred.label.lower() == example.label.lower()

    allowed_labels = get_allowed_labels(mode=mode)

    # Create the evaluator
    evaluate = dspy.Evaluate(
        devset=examples,
        metric=metric,
        num_threads=50,
        max_errors=1000,
        display_progress=True,
        display_table=True,
        return_all_scores=True,
        provide_traceback=True,
    )

    # Evaluate the model
    eval_results = evaluate(program=fact_checker)

    # Calculate metrics
    report = classification_report(
        ref_labels,
        pred_labels,
        labels=allowed_labels,
        output_dict=True,
        zero_division=np.nan,  # type: ignore
    )

    for label, r in report.items():  # type: ignore
        if isinstance(r, dict):
            for metric_name, value in r.items():
                mlflow.log_metric(f"{name}_{label}_{metric_name}", value)
        else:
            # For accuracy
            mlflow.log_metric(f"{name}_{label}", r)

    report_pd = pd.DataFrame(report).transpose()
    mlflow.log_table(report_pd, artifact_file=f"{name}_classification_report.json")

    output_path = os.path.join(output_folder, name + ".json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return eval_results, report
