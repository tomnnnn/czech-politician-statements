import mlflow
import csv
from fact_checker import FactChecker 
import os
from .utils import create_examples, evaluate, sample_statements, setup_dspy, setup_mlflow

# create examples
statements = sample_statements("random", portion=0.05)
examples = create_examples(statements)

# prepare output folder
OUTPUT_FOLDER = "results/dspy/num_hops"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

setup_mlflow("Find Best num_hops")
setup_dspy()

k = 4
num_hops_range = range(1,10)

with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["num_hops", "macro_f1"])

for num_hops in num_hops_range:
    with mlflow.start_run(run_name=str(num_hops)) as run:
        print(f"Running num_hops: {num_hops}")
        fact_checker = FactChecker(search_endpoint="http://localhost:4242/search", retrieval_hops=num_hops, per_hop_documents=k, mode="binary")
        results, report = evaluate(fact_checker, examples, OUTPUT_FOLDER, mode="binary")

    with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "a+") as f:
        writer = csv.writer(f)
        writer.writerow([num_hops, report["macro avg"]["f1-score"]])

with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["num_hops", "macro_f1"])

# create examples
statements = sample_statements("filtered", portion=0.3)
examples = create_examples(statements)

k = 2
num_hops_range = [18]
macro_f1 = []

setup_mlflow("Find Best num_hops (filtered)")
setup_dspy()

for num_hops in num_hops_range:
    with mlflow.start_run(run_name=str(num_hops)) as run:
        print(f"Running num_hops: {num_hops}")
        fact_checker = FactChecker(search_endpoint="http://localhost:4242/search", retrieval_hops=num_hops, per_hop_documents=k, mode="binary")
        results, report = evaluate(fact_checker, examples, OUTPUT_FOLDER, mode="binary")
        macro_f1.append(report["macro avg"]["f1-score"])

    with open(os.path.join(OUTPUT_FOLDER, "results.csv"), "a+") as f:
        writer = csv.writer(f)
        writer.writerow([num_hops, report["macro avg"]["f1-score"]])
