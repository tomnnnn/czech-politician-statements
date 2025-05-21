"""
This module implements a naive veracity predictor for setting a baseline performance.

@author: Hai Phong Nguyen
"""
import dspy
from dataset_manager.models import Statement
from typing import Literal

class VeracityBinary(dspy.Signature):
    """
    Given a statement, author, and date, determine whether the statement is true or false.
    Base your decision on real-world knowledge and consider carefully the context of the statement.
    When dealing with exact numbers, use 10 % tolerance except for when the statement
    itself explicitly emphasizes the exactness of the number.
    """
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()

    label: Literal["pravda", "nepravda"] = dspy.OutputField()

class NaiveVeracityPredictor(dspy.Module):
    def __init__(
        self,
        **kwargs,
    ):
        self.classify = dspy.ChainOfThought(VeracityBinary)


    def forward(self, statement: Statement) -> dspy.Prediction:
        # get evidence
        label = self.classify(
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
        ).label

        # create and return the prediction
        return dspy.Prediction(
            statement=statement.statement,
            label=label,
        )
