"""
Baseline experiments using the veracity predictor without any evidence.

@author: Hai Phong Nguyen
"""
from .utils import setup_evaluation, evaluate, sample_statements,create_examples
from fact_checker.naive import NaiveVeracityPredictor
import mlflow
from dataset_manager import Dataset

OUTPUT_FOLDER = "results/dspy/baseline_no_evidence"

samples = setup_evaluation(
    sample_portions=[1, 0.2, 0.3],
    sample_types=["random", "random", "filtered"],
    experiment_name="Baseline No Evidence",
    mode="binary",
    output_folder=OUTPUT_FOLDER,
)

dataset = Dataset("datasets/curated.sqlite")
statements = dataset.get_statements(allowed_labels=['pravda', 'nepravda'])
curated_examples = create_examples(statements)

with mlflow.start_run(run_name="Curated"):
    mlflow.log_param("num_samples", len(curated_examples))
    mlflow.log_param("sample_portion", 1)
    mlflow.log_param("curation", "manual")

    vp = NaiveVeracityPredictor()
    evaluate(vp, curated_examples, OUTPUT_FOLDER, mode="binary")

with mlflow.start_run(run_name="All"):
    mlflow.log_param("num_samples", len(samples))
    vp = NaiveVeracityPredictor()
    evaluate(vp, samples[0], OUTPUT_FOLDER, mode="binary")

with mlflow.start_run(run_name="Auto Curated"):
    mlflow.log_param("num_samples", len(samples))
    mlflow.log_param("sample_portion", 0.3)
    mlflow.log_param("curation", "auto")

    vp = NaiveVeracityPredictor()
    evaluate(vp, samples[2], OUTPUT_FOLDER, mode="binary")
