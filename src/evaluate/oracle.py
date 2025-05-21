"""
Script to evaluate FactChecker on oracle evidence to set target performance.

@author: Hai Phong Nguyen
"""
import mlflow
from fact_checker import FactChecker
from fact_checker.retrievers import SimpleRetriever
from .utils import setup_evaluation, evaluate, create_examples
from dataset_manager import Dataset

MODE = "binary"
OUTPUT_FOLDER = "results/dspy/simple_retriever"

samples = setup_evaluation(
    experiment_name="Oracle Evidence",
    sample_types=["filtered", "random"],
    sample_portions=[1, 1],
    min_num_evidence=1,
    mode=MODE,
    output_folder=OUTPUT_FOLDER,
)

dataset = Dataset("datasets/curated.sqlite")
statements = dataset.get_statements()
curated_samples = create_examples(statements)

with mlflow.start_run(run_name="Filtered"):
    mlflow.log_param("sampling_type", "cheat")
    mlflow.log_param("portion", 1)
    mlflow.log_param("label_mode", MODE)

    retriever = SimpleRetriever("datasets/other/cheat.sqlite", html=False)
    fact_checker = FactChecker(mode=MODE, link_entities=False, retriever=retriever)
    evaluate(fact_checker, samples[0], mode=MODE, output_folder=OUTPUT_FOLDER)

with mlflow.start_run(run_name="Whole Dataset"):
    mlflow.log_param("sampling_type", "random")
    mlflow.log_param("portion", 1)
    mlflow.log_param("label_mode", MODE)

    retriever = SimpleRetriever("datasets/other/cheat.sqlite", html=False)
    fact_checker = FactChecker(mode=MODE, link_entities=False, retriever=retriever)
    evaluate(fact_checker, samples[1], mode=MODE, output_folder=OUTPUT_FOLDER)

with mlflow.start_run(run_name="Curated"):
    mlflow.log_param("sampling_type", "curated")
    mlflow.log_param("portion", 1)
    mlflow.log_param("label_mode", MODE)

    retriever = SimpleRetriever("datasets/other/cheat.sqlite", html=False)
    fact_checker = FactChecker(mode=MODE, link_entities=False, retriever=retriever)
    evaluate(fact_checker, curated_samples, mode=MODE, output_folder=OUTPUT_FOLDER)
