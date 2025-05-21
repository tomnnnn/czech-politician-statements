import argparse
import json
import asyncio
import os
import logging
import dspy
from tqdm.asyncio import tqdm_asyncio
import mlflow

from dataset_manager import Dataset

from fact_checker.auto_curator import VerifiablePredictor

# mute warnings from mlflow
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)


def configure_logging(log_path):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the FactChecker model.")

    parser.add_argument("--search-base-api", type=str, default="htttp://localhost:4242", help="Base URL for the search API.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/demagog.db",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hosted_vllm/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        help="Model name or path.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="fact-checker-eval",
        help="Path to save the evaluation results.",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=4,
        help="Number of hops for the retriever.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=4,
        help="Number of documents to retrieve per hop.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        help="Classification mode for the fact checker.",
    )

    args = parser.parse_args()
    return args


async def main():
    args = parse_args()

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
    mlflow.set_experiment("FilterUnverifiable")

    os.makedirs(args.output_folder, exist_ok=True)

    with mlflow.start_run():
        lm = dspy.LM(
            args.model,
            api_base=args.api_base,
            api_key=os.environ.get("API_KEY"),
            temperature=0.0,
            max_tokens=3000,
            rpm=60
        )
        dspy.configure(lm=lm)

        dataset = Dataset(args.dataset)
        predictor = dspy.asyncify(VerifiablePredictor())
        sem = asyncio.Semaphore(200)

        async def predict_limited(statement) -> dspy.Prediction:
            async with sem:
                try:
                    return await predictor(statement=statement)
                except Exception as e:
                    return dspy.Prediction(
                        statement=statement,
                        verifiable=False,
                        evidence=[],
                        error=str(e),
                    )

        print("Gettig statements from the dataset...")
        statements = dataset.get_statements(min_evidence_count=1)

        print("Filtering statements...")
        coroutines = [
            predict_limited(statement)
            for statement in statements
        ]

        verifibilities = await tqdm_asyncio.gather(*coroutines)

        verifiables = [
            {
                "id": v.statement.id,
                "statement": v.statement.statement,
                "evidence": v.evidence,
            }
            for v in verifibilities if v.verifiable
        ]


    output_path = os.path.join(args.output_folder, "verifiable_statements.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verifiables, f, ensure_ascii=False, indent=4)

    print("Total of {} verifiable statements.".format(len(verifiables)))
    print("Results saved to:", output_path)


if __name__ == "__main__":
    asyncio.run(main())
