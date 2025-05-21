"""
This module contains the VerifiablePredictor class, which is used to determine
whether a given statement is verifiable based on the evidence provided.

@author: Hai Phong Nguyen
"""

import dspy
import json
import logging
from dataset_manager.models import Statement
from .retrievers import HopRetriever

logger = logging.getLogger(__name__)

class EnoughToSupport(dspy.Signature):
    """
    Predict whether presented evidence has enough information to support the evidence.
    """

    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()

    enough: bool = dspy.OutputField(
        description="Is the provided evidence enough to support the statement?",
    )

class EnoughToRefute(dspy.Signature):
    """
    Predict whether presented evidence has enough information to refute the evidence.
    """

    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()

    enough: bool = dspy.OutputField(
        description="Is the provided evidence enough to refute the statement?",
    )


class VerifiablePredictor(dspy.Module):
    def __init__(
        self,
        retrieval_hops=6,
        per_hop_documents=5,
        **kwargs,
    ):
        self.retriever = HopRetriever(
            num_hops=retrieval_hops, 
            num_docs=per_hop_documents,
            search_base_api=kwargs.get("search_base_api", None),
        )

        self.enough_support = dspy.ChainOfThought(EnoughToSupport)
        self.enough_refute = dspy.ChainOfThought(EnoughToRefute)


    def forward(self, statement: Statement) -> dspy.Prediction:
        # get evidence
        retriever_results = self.retriever(statement)
        evidence = retriever_results.evidence
        used_queries= retriever_results.used_queries

        # predict whether the evidence is enough to support or refute the statement
        predict_input = {
            "statement": statement.statement,
            "author": statement.author,
            "date": statement.date,
            "evidence": evidence,
        }

        if statement.label.lower() == 'pravda':
            enough = self.enough_support(**predict_input)
            verifiable = enough.enough
        elif statement.label.lower() == 'nepravda':
            enough = self.enough_refute(**predict_input)
            verifiable = enough.enough
        else:
            verifiable = False

        # create and return the prediction
        return dspy.Prediction(
            statement=statement,
            evidence=evidence,
            verifiable=verifiable,
            used_queries=used_queries,
        )
