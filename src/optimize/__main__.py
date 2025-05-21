"""
Script to optimize prompts of the fact-checker using MIPROv2 and evaluate the results.

@author: Hai Phong Nguyen
"""

import json
import logging
import os
from typing import Literal

import dspy
import mlflow
import numpy as np
import pandas as pd
from dataset_manager import Dataset
from dspy.teleprompt import MIPROv2, SIMBA
from dataset_manager.models import Statement
from fact_checker.fact_checker import FactChecker
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# mute warnings from mlflow
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)


def evaluate(
    fact_checker: dspy.Module,
    examples: list[dspy.Example],
    output_folder: str,
    allowed_labels: list[str],
    name: str = "evaluation",
):
    """
    Evaluate the fact checker on the provided examples and saves the results.
    """

    pred_labels = []
    ref_labels = []

    def metric(example, pred, trace=None):
        pred_labels.append(pred.label.lower())
        ref_labels.append(example.label.lower())

        return pred.label.lower() == example.label.lower()

    # Create the evaluator
    evaluate = dspy.Evaluate(
        devset=examples,
        metric=metric,
        num_threads=20,
        max_errors=100,
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


def optimize(
    lm,
    fact_checker: dspy.Module,
    train: list[dspy.Example],
    output_folder: str,
    seed: int = 42,
):
    """
    Optimize the fact checker in zeroshot settings using MIPROv2.
    """

    def metric(example, pred, trace=None):
        return pred.label.lower() == example.label.lower()

    teleprompter = MIPROv2(
        metric=metric,
        auto="medium",
        prompt_model=lm,
        teacher_settings=dict(lm=lm),
        max_errors=100,
    )

    zeroshot_optimized = teleprompter.compile(
        fact_checker.deepcopy(),
        trainset=train,
        seed=seed,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        requires_permission_to_run=False,
    )

    # Save the optimized model
    output_path = os.path.join(output_folder, "optimized_model.pkl")
    zeroshot_optimized.save(output_path)

    return zeroshot_optimized


def optimize_simba(
    fact_checker: dspy.Module,
    train: list[dspy.Example],
    output_folder: str,
    seed: int = 42,
):
    """
    Optimize the fact checker in zeroshot settings using MIPROv2.
    """

    def metric(example, pred, trace=None):
        return pred.label.lower() == example.label.lower()

    teleprompter = SIMBA(metric=metric, max_steps=10, max_demos=10)

    zeroshot_optimized = teleprompter.compile(
        fact_checker.deepcopy(),
        trainset=train,
        seed=seed,
    )

    os.makedirs(output_folder, exist_ok=True)
    # Save the optimized model
    output_path = os.path.join(output_folder, "optimized_model.pkl")
    zeroshot_optimized.save(output_path)

    return zeroshot_optimized


def sample_statements(
    allowed_labels,
    type: Literal["random", "autocurated"] = "random",
    train_portion: float = 0.2,
):
    dataset = Dataset(DATASET_PATH)
    statements = dataset.get_statements(allowed_labels=allowed_labels)
    statements = [s for s in statements if s.label.lower() in allowed_labels]

    if type == "autocurated":
        with open("fact-checker-eval/verifiable_statements.json", "r") as f:
            verifiable_statements = json.load(f)

        verifiable_ids = [item["id"] for item in verifiable_statements]
        statements = [s for s in statements if s.id in verifiable_ids]

    labels = [s.label for s in statements]
    train_statements, dev_statements = train_test_split(
        statements,
        train_size=train_portion,
        stratify=labels,
        random_state=42,
    )

    # report label distribution
    label_distribution = {
        label: len(
            [s for s in train_statements + dev_statements if s.label.lower() == label]
        )
        for label in allowed_labels
    }
    print("Label distribution:")
    for label, count in label_distribution.items():
        print(f"{label}: {count}")

    return train_statements, dev_statements

def create_examples(test: list[Statement], train: list[Statement]):
    trainset = [
        dspy.Example(
            statement=statement,
            label=statement.label,
        ).with_inputs("statement", "search_func")
        for statement in train
    ]

    devset = [
        dspy.Example(
            statement=statement,
            label=statement.label,
        ).with_inputs("statement", "search_func")
        for statement in test
    ]

    return trainset, devset


NUM_HOPS = 2
NUM_DOCS = 4
DATASET_PATH = "datasets/dataset_demagog.sqlite"
OUTPUT_DIR = "results/optimized_fact_checker"
ALLOWED_LABELS = ["pravda", "nepravda"]

def main():
    mlflow.set_tracking_uri("sqlite:///thesis_runs.db")
    mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
    mlflow.set_experiment("Prompt Optimization")

    with mlflow.start_run(run_name="MIPROv2 Auto Curated"):
        lm = dspy.LM(
            "hosted_vllm/Qwen/Qwen2.5-32B-Instruct-AWQ",
            api_base="http://localhost:8000/v1",
            api_key=os.environ.get("API_KEY"),
            temperature=0.0,
            max_tokens=3000,
        )
        dspy.configure(lm=lm)

        fact_checker = FactChecker(
            retrieval_hops=NUM_HOPS, per_hop_documents=NUM_DOCS, mode="binary"
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        train_statements, dev_statements = sample_statements(ALLOWED_LABELS, "autocurated", 0.2)
        train_examples, dev_examples = create_examples(dev_statements, train_statements)

        mlflow.log_param("num_train", len(train_statements))
        mlflow.log_param("num_dev", len(dev_statements))
        mlflow.log_param("num_hops", NUM_HOPS)
        mlflow.log_param("num_docs", NUM_DOCS)
        mlflow.log_param("allowed_labels", ALLOWED_LABELS)
        mlflow.log_param("model", "Qwen/Qwen2.5-32B-Instruct-AWQ")

        label_distribution = {
            label: len(
                [
                    s
                    for s in train_statements + dev_statements
                    if s.label.lower() == label
                ]
            )
            for label in ALLOWED_LABELS
        }

        for label, count in label_distribution.items():
            mlflow.log_param(f"label_{label}", count)

        print("Optimizing the fact checker...")
        # optimized = optimize(lm, fact_checker,train_examples, OUTPUT_DIR, seed=42)
        fact_checker.load("fact-checker-eval/best_model.json")

        # Evaluate the optimized fact checker
        print("Post-optimization evaluation...")
        evaluate(
            fact_checker,
            dev_examples,
            OUTPUT_DIR,
            ALLOWED_LABELS,
            "post_optimize_evaluation",
        )

    print("Prompt optimization completed.")
    print("Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
