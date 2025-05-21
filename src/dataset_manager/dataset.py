"""
This module provides a Dataset class for managing a database of statements, articles, and segments.

@author: Hai Phong Nguyen
"""

from collections import defaultdict
import datetime
import os
from .orm import *
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import NoResultFound

class Dataset:
    def __init__(self, path: str, create_if_not_exists: bool = False, read_only: bool = False):
        if path != ":memory:" and not create_if_not_exists and not os.path.exists(path):
            raise FileNotFoundError(f"Database file not found: {path}")

        self.session = init_db(path, read_only)

    def _filter_column_keys(self, data, model) -> dict:
        """
        Filters the keys of the data dictionary to only include those that are present in the model's columns.
        """
        column_names = {column.name for column in model.__table__.columns}
        return {key: value for key, value in data.items() if key in column_names}

    def get_statements(self, allowed_labels=None, min_evidence_count=0):
        statements = self.session.query(Statement)

        if allowed_labels:
            statements = statements.filter(func.lower(Statement.label).in_(allowed_labels))

        if min_evidence_count > 0:
            statements = statements.join(ArticleRelevance).group_by(Statement.id).having(func.count(ArticleRelevance.article_id) >= min_evidence_count)

        return statements.all()

    def get_statement(self, statement_id) -> Statement|None:
        return (self.session.query(Statement).filter(Statement.id == statement_id).first())

    def get_articles(self, statement_id = None) -> list[Article]:
        articles = self.session.query(Article)

        if statement_id:
            articles = articles.join(ArticleRelevance).filter(ArticleRelevance.statement_id == statement_id)

        return articles.all()


    def get_article(self, article_id) -> Article|None:
        return (self.session.query(Article).filter(Article.id == article_id).first())


    def get_segments_with_statement_ids(self, article_id=None) -> list[tuple[int, Segment]]:
        segments = (self.session.query(ArticleRelevance.statement_id,Segment)
                    .select_from(Segment)
                    .join(ArticleRelevance, Segment.article_id == ArticleRelevance.article_id))


        if article_id:
            segments = segments.filter(Segment.article_id == article_id)

        segments = segments.all()

        # convert to list of tuples
        return [(statement_id, segment) for statement_id, segment in segments]

    def get_segments(self, article_id=None) -> list[Segment]:
        segments = self.session.query(Segment)

        if article_id:
            segments = segments.filter(Segment.article_id == article_id)

        return segments.all()

    def get_segments_by_statements(self, statement_ids: list[int]) -> dict[int, list[Segment]]:
        """
        Get all segments belonging to articles that are relevant to a statement.
        """
        # Query to fetch all segments linked to statements through articles
        segments = (
            self.session.query(ArticleRelevance.statement_id, Segment)
            .select_from(ArticleRelevance)
            .join(Article, ArticleRelevance.article_id == Article.id)
            .join(Segment, Segment.article_id == Article.id)
            .filter(ArticleRelevance.statement_id.in_(statement_ids))
        ).all()

        # Process the results into a dictionary
        statement_segments = defaultdict(list)

        for statement_id, segment in segments:
            statement_segments[statement_id].append(segment)

        statement_segments = dict(statement_segments)

        return statement_segments

    def get_segments_by_statement(self, statement_id: int) -> list[Segment]:
        """
        Get all segments belonging to articles that are relevant to a statement.
        """
        # Query to fetch all segments linked to statements through articles
        segments = (
            self.session.query(Segment)
            .select_from(ArticleRelevance)
            .join(Article, ArticleRelevance.article_id == Article.id)
            .join(Segment, Segment.article_id == Article.id)
            .filter(ArticleRelevance.statement_id == statement_id)
        ).all()

        return segments

    def get_segment(self, segment_id) -> Segment|None:

        return (self.session.query(Segment).filter(Segment.id == segment_id).first())

    def get_segment_relevances(self, statement_id=None, segment_id=None):
        relevances = self.session.query(SegmentRelevance)

        if statement_id:
            relevances = relevances.filter(SegmentRelevance.statement_id == statement_id)

        if segment_id:
            relevances = relevances.filter(SegmentRelevance.segment_id == segment_id)

        return relevances.all()


    def get_article_relevances(self, statement_id=None, article_id=None) -> list[ArticleRelevance]:
        relevances = self.session.query(ArticleRelevance)

        if statement_id:
            relevances = relevances.filter(ArticleRelevance.statement_id == statement_id)

        if article_id:
            relevances = relevances.filter(ArticleRelevance.article_id == article_id)

        return relevances.all()


    def set_segment_relevance(self, segment_id: int, statement_id: int, relevance: float) -> None:
        """
        Sets or updates the relevance of a segment to a statement.
        """
        try:
            rel = self.session.query(SegmentRelevance).filter_by(
                segment_id=segment_id,
                statement_id=statement_id
            ).one()
        except NoResultFound:
            rel = SegmentRelevance(
                segment_id=segment_id,
                statement_id=statement_id,
                relevance=relevance
            )
            self.session.add(rel)

        rel.relevance = relevance
        self.session.commit()


    def set_article_relevance(self, article_id, statement_id) -> None:
        """
        Sets or updates the relevance of a segment to a statement.
        """
        self.session.add(ArticleRelevance(
            article_id=article_id,
            statement_id=statement_id,
        ))
        self.session.commit()


    def delete_statement(self, statement_id) -> None:
        self.session.delete(self.session.query(Statement).filter(Statement.id == statement_id).first())
        self.session.commit()

    def delete_article(self, article_id) -> None:
        self.session.delete(self.session.query(Article).filter(Article.id == article_id).first())
        self.session.commit()

    def delete_segment(self, segment_id) -> None:
        self.session.delete(self.session.query(Segment).filter(Segment.id == segment_id).first())
        self.session.commit()

    def insert_statement(self, statement: dict[str, str]) -> Statement:
        statement = self._filter_column_keys(statement, Statement)

        new_statement = Statement( **statement,)

        self.session.add(new_statement)
        self.session.commit()

        return new_statement

    def insert_statements(self, statements) -> list[Statement]:
        statements = [self._filter_column_keys(statement, Statement) for statement in statements]
        new_statements = [
            Statement( **statement,)
            for statement in statements
        ]

        self.session.add_all(new_statements)
        self.session.commit()
        return new_statements

    def insert_article(self, article: dict) -> Article:
        article = self._filter_column_keys(article, Article)

        new_article = Article(**article)

        self.session.add(new_article)
        self.session.commit()
        return new_article

    def insert_articles(self, articles: list[dict]) -> list[Article]:
        articles = [self._filter_column_keys(article, Article) for article in articles]

        new_articles = [
            Article(**article)
            for article in articles
        ]

        self.session.add_all(new_articles)
        self.session.commit()
        return new_articles

    def insert_segment(self, segment: dict) -> Segment:
        segment = self._filter_column_keys(segment, Segment)
        new_segment = Segment(**segment)

        self.session.add(new_segment)
        self.session.commit()
        return new_segment

    def insert_segments(self, segments: list[dict]) -> list[Segment]:
        segments = [self._filter_column_keys(segment, Segment) for segment in segments]

        new_segments = [ Segment(**segment) for segment in segments ]

        self.session.add_all(new_segments)
        self.session.commit()

        return new_segments
    
    def statements_count(self):
        return self.session.query(Statement).count()

    def articles_count(self):
        return self.session.query(Article).count()

    def segment_count(self):
        return self.session.query(Segment).count()
