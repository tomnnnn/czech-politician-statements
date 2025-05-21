"""
Dummy retriever that returns an empty list of evidence.

@author: Hai Phong Nguyen
"""

import dspy

from dataset_manager.models import Statement

class SimpleRetriever(dspy.Module):
    def __init__(self, num_docs=3):
        self.num_docs = num_docs

    async def forward(self, statement: Statement) -> dspy.Prediction:
        return dspy.Prediction(
            evidence = [],
        )

