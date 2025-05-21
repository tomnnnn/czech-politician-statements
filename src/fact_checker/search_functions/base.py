"""
Abstract base class for evidence search functions, used to 
search for relevant segments in evidence articles inside the dataset.

@author: Hai Phong Nguyen
"""

from abc import ABC, abstractmethod
from typing import Optional

import faiss
from dataset_manager.models import Segment


class SearchFunction(ABC):
    @abstractmethod
    async def add_index(
        self,
        segments: list[Segment],
        save_path: Optional[str],
        load_if_exists: bool,
        save: bool,
        key: str | int = "_default",
    ):
        """
        Creates or loads an index and adds it to internal indices dictionary.

        Args:
            segments (list[Segment]): List of segments to index.
            key (str): Key for the index.
            save_path (Optional[str]): Path to save the index.
            load (Union[Literal["auto"], bool]): Whether to load the index if it exists.
        """
        pass

    @abstractmethod
    def search(
        self, query: str, k: int = 10, key: str | int = "_default"
    ) -> list[Segment]:
        pass

    @abstractmethod
    async def search_async(
        self, query: str, k: int = 10, key: str | int = "_default"
    ) -> list[Segment]:
        pass
