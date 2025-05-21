"""
This script implements a simple naive retriever which returns all evidence
articles related to a statement, without any modifications.

@author: Hai Phong Nguyen
"""

import dspy
from dataset_manager import Dataset
from bs4 import BeautifulSoup

from dataset_manager.models import Statement

class SimpleRetriever(dspy.Module):
    def __init__(self, dataset_path: str = "datasets/dataset_demagog.sqlite", num_docs=4, html=True):
        self.num_docs = num_docs
        self.dataset = Dataset(dataset_path)
        self.html = html

    def extract_article_text(self, article):
        """
        Cleanup the HTML of the article and extract the text from key content tags.
        """
        soup = BeautifulSoup(article.content, 'html.parser')
        
        # Tags likely to contain meaningful article text
        tags_to_extract = ['h1', 'h2', 'h3', 'p', 'li', 'blockquote']
        
        elements = soup.find_all(tags_to_extract)

        text = "\n".join([el.get_text(strip=True) for el in elements if len(el) > 30])
        return text

    def forward(self, statement: Statement) -> dspy.Prediction:
        query = statement.statement
        articles = self.dataset.get_articles(statement.id)
        documents = []

        for article in articles:
            documents.append({
                "title": article.title,
                "text": self.extract_article_text(article) if self.html else article.content,
                "url": article.source,
            })


        return dspy.Prediction(
            evidence = documents,
            used_queries = [query],
            info=""
        )

