"""
Script to evaluate FactChecker on full-text evidence articles.

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
    experiment_name="Simple Retriever",
    sample_types=["random", "filtered"],
    sample_portions=[0.2, 1],
    min_num_evidence=1,
    mode=MODE,
    output_folder=OUTPUT_FOLDER,
)

dataset = Dataset("datasets/curated.sqlite")
statements = dataset.get_statements(allowed_labels=['pravda', 'nepravda'])
curated_examples = create_examples(statements)


with mlflow.start_run(run_name="Curated"):
    mlflow.log_param("portion", 1)
    mlflow.log_param("label_mode", MODE)
    mlflow.log_param("sampling_type", "curated")

    retriever = SimpleRetriever()
    fact_checker = FactChecker(mode=MODE, link_entities=False, retriever=retriever)
    evaluate(fact_checker, curated_examples, mode=MODE, output_folder=OUTPUT_FOLDER)

with mlflow.start_run(run_name="Random Sample"):
    mlflow.log_param("sampling_type", "random")
    mlflow.log_param("portion", 0.2)
    mlflow.log_param("label_mode", MODE)

    retriever = SimpleRetriever()
    fact_checker = FactChecker(mode=MODE, link_entities=False, retriever=retriever)
    evaluate(fact_checker, samples[0], mode=MODE, output_folder=OUTPUT_FOLDER)


with mlflow.start_run(run_name="Auto Curated"):
    mlflow.log_param("sampling_type", "filtered")
    mlflow.log_param("portion", 1)
    mlflow.log_param("label_mode", MODE)

    retriever = SimpleRetriever()
    fact_checker = FactChecker(mode=MODE, link_entities=False, retriever=retriever)
    evaluate(fact_checker, samples[1], mode=MODE, output_folder=OUTPUT_FOLDER)


print("************************* DONT FORGET TO DECOMMENT FACT CHECKER INFO *******************************")
