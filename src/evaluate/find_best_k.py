"""
Script to empirically find best number of retrieved documents per hop in FactChecker with HopRetriever.
"""

import mlflow
from fact_checker import FactChecker

from .utils import evaluate, setup_evaluation

MODE = "binary"
OUTPUT_FOLDER = "results/dspy/find_best_k"
NUM_HOPS = 7

random_sample = setup_evaluation(
    experiment_name="Find best k for HopRetriever",
    sample_types=["filtered"],
    sample_portions=[0.3],
    mode=MODE,
    output_folder=OUTPUT_FOLDER,
)

k_range = [2]
best_k = (0, 0)

for k in k_range:
    with mlflow.start_run(run_name=f"{k}"):
        mlflow.log_param("sample_type", "filtered")
        mlflow.log_param("sample_portion", 0.3)
        mlflow.log_param("mode", MODE)
        mlflow.log_param("num_hops", NUM_HOPS)
        mlflow.log_param("k", k)

        fact_checker = FactChecker(
            mode=MODE, link_entities=False, per_hop_documents=k, retrieval_hops=NUM_HOPS
        )
        _, report = evaluate(fact_checker, random_sample[0], mode=MODE, output_folder=OUTPUT_FOLDER)
        best_k = (k, report["macro avg"]["f1-score"]) if report["macro avg"]["f1-score"] > best_k[1] else best_k

exit(best_k[0])
