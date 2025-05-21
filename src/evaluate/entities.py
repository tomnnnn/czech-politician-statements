"""
Script to evaluate the performance of the FactChecker on automatically curated dataset.
"""
import mlflow
from fact_checker import FactChecker
from .utils import setup_evaluation, evaluate, create_examples
from dataset_manager import Dataset

MODE = "binary"
OUTPUT_FOLDER = "results/dspy/auto_curated"

examples = setup_evaluation(
    experiment_name="Multi-Hop Hybrid With Entity Linking",
    sample_types=["filtered", "random"],
    sample_portions=[1, 1],
    mode=MODE,
    output_folder=OUTPUT_FOLDER,
    min_num_evidence=1,
)

dataset = Dataset("datasets/curated.sqlite")
statements = dataset.get_statements(allowed_labels=['pravda', 'nepravda'])
curated_examples = create_examples(statements)

with mlflow.start_run(run_name="Auto Curated"):
    fact_checker = FactChecker(mode=MODE, link_entities=True)
    evaluate(fact_checker, examples[0], mode=MODE, output_folder=OUTPUT_FOLDER)

with mlflow.start_run(run_name="Whole Dataset"):
    fact_checker = FactChecker(mode=MODE, link_entities=True)
    evaluate(fact_checker, examples[1], mode=MODE, output_folder=OUTPUT_FOLDER)

with mlflow.start_run(run_name="Curated Dataset"):
    mlflow.log_param("sample_type", "curated")
    mlflow.log_param("sample_portion", 1)
    mlflow.log_param("mode", MODE)

    fact_checker = FactChecker(mode=MODE, link_entities=True)
    evaluate(fact_checker, curated_examples, mode=MODE, output_folder=OUTPUT_FOLDER)
