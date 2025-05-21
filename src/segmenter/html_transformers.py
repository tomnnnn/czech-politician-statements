"""
This module contains classes for transforming HTML elements into text formats.

@author: Hai Phong Nguyen
"""

import csv
import io
from abc import ABC, abstractmethod
from semantic_router.encoders import HuggingFaceEncoder
from semantic_chunkers import StatisticalChunker

def normalize_text(text: str) -> str:
    """
    Normalize text by removing multiple whitespaces and newlines. Only single newline or space is preserved.
    """
    return " ".join(text.split())

class ElementTransformer(ABC):
    @abstractmethod
    def transform(self, el) -> list[str]:
        pass

class Paragraph(ElementTransformer):
    def __init__(self):
        encoder=HuggingFaceEncoder(name="BAAI/bge-m3", batch_size=1024)
        self.chunker = StatisticalChunker(encoder=encoder, max_split_tokens=250)

    def transform(self, el) -> list[str]:
        text = normalize_text(el.text.strip())

        if len(text) > 6000:
            print("Paragraph too long, chunking using semantic chunker")
            # Split long paragraphs into smaller segments using semantic chunker
            chunks = self.chunker(docs=[text])
            segments = [chunk.content for chunk in chunks[0]]
            print(segments[0] + "\n\n")
            return segments
            
        else:
            return [text]

class List(ElementTransformer):
    def transform(self, el) -> list[str]:
        """
        Transforms <ol> and <ul> elements into formatted text.
        Supports nested lists and maintains correct formatting.
        """
        return self._parse_list(el)

    def _parse_list(self, el, level=1, prefix="") -> list[str]:
        items = []
        is_ordered = el.name == "ol"

        for i, li in enumerate(el.find_all("li", recursive=False), start=1):
            # Determine prefix: numbered for <ol>, bullet for <ul>
            item_prefix = f"{prefix}{i}. " if is_ordered else f"{prefix}- "

            # Extract text while preserving indentation
            li_text = li.get_text().strip()

            # Process nested lists, if any
            nested_list = li.find(["ul", "ol"])
            if nested_list:
                nested_text = self._parse_list(nested_list, level + 1, prefix + "   ")
                items.append(f"{item_prefix}{li_text}\n{nested_text}")
            else:
                items.append(f"{item_prefix}{li_text}")

        return ["\n".join(items)]

class Table(ElementTransformer):
    def transform(self, el) -> list[str]:
        """
        Transform table to csv string
        """
        rows = []
        for row in el.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
            rows.append(cells)

        # Convert to CSV string
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(rows)

        csv_string = output.getvalue()
        output.close()        

        return [csv_string]

