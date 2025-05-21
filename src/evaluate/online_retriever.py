import os

import mlflow

from fact_checker import FactChecker

from .utils import evaluate, setup_evaluation

# prepare output folder
output_folder = "results/dspy/online_retriever"
examples = setup_evaluation(
    output_folder=output_folder,
    experiment_name="Online Retriever",
    sample_portions=[0.01],
    sample_types=["random"]
)

with mlflow.start_run(run_name="Online Retriever") as run:
    mlflow.log_param("num_samples", len(examples[0]))
    mlflow.log_param("sample_portion", 0.2)
    mlflow.log_param("sample_type", "random")
    mlflow.log_param("num_hops", 4)
    mlflow.log_param("k", 4)

    fact_checker = FactChecker(search_endpoint="http://localhost:4242/search-online", retrieval_hops=4, k=4, mode='binary')
    results = evaluate(fact_checker, examples[0], output_folder, mode="binary")


with mlflow.start_run(run_name="Offline Retriever") as run:
    mlflow.log_param("num_samples", len(examples[0]))
    mlflow.log_param("sample_portion", 0.01)
    mlflow.log_param("sample_type", "random")
    mlflow.log_param("num_hops", 4)
    mlflow.log_param("k", 4)

    fact_checker = FactChecker(search_endpoint="http://localhost:4242/search", retrieval_hops=4, k=4, mode='binary')
    results = evaluate(fact_checker, examples[0], output_folder, mode="binary")

