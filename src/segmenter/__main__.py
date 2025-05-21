"""
This script segments articles into smaller segments and inserts them into a dataset.

@author: Hai Phong Nguyen
"""

from dataset_manager import Dataset
from .segment import segment_article
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment articles into smaller segments.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file", required=True)
    args = parser.parse_args()

    dataset = Dataset(args.dataset_path)
    articles = dataset.get_articles()

    segments = []
    for _,article in enumerate(tqdm.tqdm(articles, desc="Segmenting articles", unit="article")):
        segments.extend(
            {"article_id": article.id, "text": segment} 
            for segment in segment_article(article.content, 50)
        )

    dataset.insert_segments(segments)
    print(f"Inserted {len(segments)} segments into the dataset.")
