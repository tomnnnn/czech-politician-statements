"""
This module is responsible for transforming HTML content into a list of string segments.

@author: Hai Phong Nguyen
"""

from .html_transformers import Paragraph, Table, List
from bs4 import BeautifulSoup

transformation_rules = {
    "p": Paragraph(),
    "table": Table(),
    "ul": List(),
    "ol": List()
}


def transform_element(el) -> list[str]:
    """
    Transform HTML element to string segment
    """
    tag = el.name
    if tag in transformation_rules:
        return transformation_rules[tag].transform(el)
    else:
        return []


def segment_article(html: str, min_len = 25) -> list[str]:
    """
    transform html to list of string segments
    """
    soup = BeautifulSoup(html, "html.parser")

    segments = [transform_element(el) for el in soup.find_all()]

    segments = [item for sublist in segments for item in sublist]

    segments = [seg for seg in segments if len(seg) > min_len]

    return segments

