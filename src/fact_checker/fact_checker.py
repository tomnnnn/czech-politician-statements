"""
Fact-checking module for verifying statements using a retriever and a classifier.
This module uses a retriever to gather evidence and a classifier to determine the veracity of a statement.
The classifier can operate in two modes: ternary (supported, refuted, unverifiable) and binary (supported, refuted).

@author: Hai Phong Nguyen
"""

import dspy
import json
from typing import Literal
from dataset_manager.models import Statement
from .retrievers import HopRetriever
import mlflow

class VeracityTernary(dspy.Signature):
    """
    Based on the given evidence and previous extracted knowledge (info), determine whether the statement is supported or refuted.
    If not enough evidence is available, classify the statement as unverifiable.
    When dealing with exact numbers, use 10 % tolerance except for when the statement
    itself explicitly emphasizes the exactness of the number. 
    """
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()
    info: list[str] = dspy.InputField(
        description="List of noted informations from the evidence during the retrieval process."
    )

    label: Literal["pravda", "nepravda", "neověřitelné"] = dspy.OutputField()

class VeracityBinary(dspy.Signature):
    """
    Based on the given evidence and previous extracted knowledge (info), determine whether the statement is supported or refuted.
    When dealing with exact numbers, use 10 % tolerance except for when the statement
    itself explicitly emphasizes the exactness of the number.
    """
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()
    # info: list[str] = dspy.InputField(
    #     description="List of noted informations from the evidence during the retrieval process."
    # )

    label: Literal["pravda", "nepravda"] = dspy.OutputField()


class FactChecker(dspy.Module):
    def __init__(
        self,
        retrieval_hops=4,
        per_hop_documents=4,
        search_endpoint="http://localhost:4242/search",
        mode: Literal["ternary", "binary"] = "ternary",
        link_entities: bool = False,
        retriever: dspy.Module|None = None,
        **kwargs,
    ):
        self.retriever = HopRetriever(
            num_hops=retrieval_hops, 
            num_docs=per_hop_documents,
            search_endpoint=search_endpoint,
            link_entities=link_entities,
        ) if retriever is None else retriever

        veracity = VeracityTernary if mode == "ternary" else VeracityBinary

        self.classify = dspy.ChainOfThought(veracity)

        mlflow.log_params({
            "fact_checker": "iterative_hop",
            "retrieval_hops": retrieval_hops,
            "per_hop_documents": per_hop_documents,
        })


    def forward(self, statement: Statement) -> dspy.Prediction:
        # get evidence
        retriever_result = self.retriever(statement)
        evidence = retriever_result.evidence
        used_queries = retriever_result.used_queries
        # info = retriever_result.info

        # classify
        label = self.classify(
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
            evidence=json.dumps(evidence, ensure_ascii=False),
            # info=info,
        ).label

        # create and return the prediction
        return dspy.Prediction(
            metadata={"retriever": "hop_retriever"},
            statement=statement.statement,
            evidence=evidence,
            label=label,
            used_queries=used_queries
        )
