import pytest

from dataset_manager import *

@pytest.fixture
def dataset():
    ds = Dataset(":memory:")
    return ds

def test_insert_and_get_statement(dataset: Dataset):
    dataset.insert_statement({
        "statement": "Cats are mammals.",
        "label": "true",
        "author": "John DoeJohn Doe",
        "date": "2020-01-01",
        "party": "Party A",
        "explanation": "Basic biology.",
        "explanation_brief": "Animals.",
        "origin": "Wikipedia"
    })

    statements = dataset.get_statements()
    assert len(statements) == 1
    assert statements[0].statement == "Cats are mammals."

def test_insert_and_get_article(dataset):
    dataset.insert_article({
        "url": "http://example.com",
        "title": "Example Title",
        "description": "Some desc",
        "content": "Full content",
        "type": "News",
        "author": "John DoeJane",
        "source": "Example News",
        "published": "2020-01-01",
    })

    articles = dataset.get_articles()
    assert len(articles) == 1
    assert articles[0].title == "Example Title"

def test_get_statements_allowed_labels(dataset):
    dataset.insert_statement({
        "statement": "Cats are mammals.",
        "label": "true",
        "author": "John DoeJohn Doe",
        "date": "2020-01-01",
        "party": "Party A",
        "explanation": "Basic biology.",
        "explanation_brief": "Animals.",
        "origin": "Wikipedia"
    })

    dataset.insert_statement({
        "statement": "Dogs are mammals.",
        "label": "true",
        "author": "John DoeJohn Doe",
        "date": "2020-01-01",
        "party": "Party B",
        "explanation": "Basic biology.",
        "explanation_brief": "Animals.",
        "origin": "Wikipedia"
    })

    dataset.insert_statement({
        "statement": "Cats are reptiles.",
        "label": "false",
        "author": "John DoeJane Doe",
        "date": "2020-01-02",
        "party": "Party B",
        "explanation": "",
        "explanation_brief": "",
        "origin": ""
    })

    allowed_labels = ["true"]
    statements = dataset.get_statements(allowed_labels=allowed_labels)

    assert len(statements) == 2
    assert statements[0].label == "true"


def test_insert_and_get_segment(dataset):
    dataset.insert_article({"url": "", "title": "", "content": "", "description": "", "type": "", "author": "John Doe", "source": "", "published": ""})
    article_id = dataset.get_articles()[0].id
    dataset.insert_segment({"article_id": article_id, "text": "Some paragraph."})
    
    segments = dataset.get_segments()
    assert len(segments) == 1
    assert segments[0].text == "Some paragraph."

def test_segment_relevance(dataset):
    # Setup
    dataset.insert_article({"url": "", "title": "", "content": "", "description": "", "type": "", "author": "John Doe", "source": "", "published": ""})
    article_id = dataset.get_articles()[0].id
    dataset.insert_segment({"article_id": article_id, "text": "Segment A"})
    dataset.insert_statement({"statement": "Claim", "label": "true", "author": "John Doe", "date": "2020-01-01", "party": "Party A", "explanation": "Basic biology.", "explanation_brief": "Animals.", "origin": "Wikipedia"})
    
    segment_id = dataset.get_segments()[0].id
    statement_id = dataset.get_statements()[0].id

    # Set relevance
    dataset.set_segment_relevance(segment_id, statement_id, 0.85)
    rel = dataset.get_segment_relevances(segment_id=segment_id, statement_id=statement_id)[0]
    assert rel.relevance == 0.85

    # Update relevance
    dataset.set_segment_relevance(segment_id, statement_id, 0.25)
    rel = dataset.get_segment_relevances(segment_id=segment_id, statement_id=statement_id)[0]
    assert rel.relevance == 0.25

def test_article_relevance(dataset):
    dataset.insert_article({"url": "", "title": "", "content": "", "description": "", "type": "", "author": "John Doe", "source": "", "published": ""})
    dataset.insert_statement({"statement": "Claim", "label": "true", "author": "John Doe", "date": "2020-01-01", "party": "Party A", "explanation": "Basic biology.", "explanation_brief": "Animals.", "origin": "Wikipedia"})

    article_id = dataset.get_articles()[0].id
    statement_id = dataset.get_statements()[0].id

    dataset.set_article_relevance(article_id, statement_id)

    rel = dataset.get_article_relevances(article_id=article_id, statement_id=statement_id)
    assert len(rel) == 1

def test_deletions(dataset):
    dataset.insert_article({"url": "", "title": "", "content": "", "description": "", "type": "", "author": "John Doe", "source": "", "published": ""})
    article_id = dataset.get_articles()[0].id
    dataset.delete_article(article_id)
    assert dataset.get_article(article_id) is None

    dataset.insert_statement({"statement": "Claim", "label": "true", "author": "John Doe", "date": "2020-01-01", "party": "Party A", "explanation": "Basic biology.", "explanation_brief": "Animals.", "origin": "Wikipedia"})
    statement_id = dataset.get_statements()[0].id
    dataset.delete_statement(statement_id)
    assert dataset.get_statement(statement_id) is None

    dataset.insert_article({"url": "", "title": "", "content": "", "description": "", "type": "", "author": "John Doe", "source": "", "published": ""})
    article_id = dataset.get_articles()[0].id
    dataset.insert_segment({"article_id": article_id, "text": "Segment text"})
    segment_id = dataset.get_segments()[0].id
    dataset.delete_segment(segment_id)
    assert dataset.get_segment(segment_id) is None
