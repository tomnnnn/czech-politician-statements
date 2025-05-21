from dataset_manager.orm import *
from sqlalchemy import insert
from collections import defaultdict
import re

dataset = init_db("../datasets/whole_cheat.sqlite")

statements = dataset.query(Statement).all()

def remove_verdict_sentences(text):
    # Define regex patterns to remove sentences containing specific verdicts
    verdict_patterns = [
        r'\bpravd[aouyiěéím]?\b',       # Matches pravda, pravdou, pravdy, pravdě, pravdím,...
        r'\bnepravd[aouyiěéím]?\b',     # Matches nepravda, nepravdou, nepravdy, nepravdě,...
        r'\bzavádějíc[íeíhoím]?\b',     # Matches zavádějící, zavádějícího, zavádějícímu, zavádějícím,...
        r'\bneověřiteln[ýáéíouě]?\b',   # Matches neověřitelný, neověřitelná, neověřitelné,...
        r'\b(?:pravdivý|nepravdivý|zavádějící|neověřitelný)\b'
    ]

    verdict_regex = re.compile('|'.join(verdict_patterns), re.IGNORECASE)
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Separate removed and kept sentences
    removed_sentences = [s for s in sentences if verdict_regex.search(s)] # for debugging
    kept_sentences = [s for s in sentences if not verdict_regex.search(s)]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(kept_sentences)
    
    return cleaned_text


cnter = 0
article_map = defaultdict(list)
for statement in statements:
    explanation = statement.explanation or statement.explanation_brief

    filtered_explanation = remove_verdict_sentences(explanation)
    new_article = Article(
        id = cnter,
        url = "",
        title = "",
        description = "",
        content = filtered_explanation,
        type = "",
        author = "",
        source = "",
        published = "",
        accessed = ""
    )

    cnter += 1

    article_map[statement.id].append(new_article)


articles = [article for article_list in article_map.values() for article in article_list]
dataset.add_all(articles)

relevance_stmt = insert(ArticleRelevance).values([{
    "statement_id": statement_id,
    "article_id": article.id,
} for statement_id, articles in article_map.items() for article in articles])

dataset.execute(relevance_stmt)
dataset.commit()

