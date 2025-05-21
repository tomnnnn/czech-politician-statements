"""
This module defines the SQLAlchemy ORM models for the dataset db schema.

@author: Hai Phong Nguyen
"""

from sqlalchemy import create_engine, Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base, sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy import Index, func

Base = declarative_base()

class ArticleRelevance(Base):
    __tablename__ = "article_relevance"
    article_id = mapped_column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    statement_id = mapped_column(Integer, ForeignKey("statements.id", ondelete="CASCADE"), primary_key=True)

    __table_args__ = (
        Index("ix_article_relevance_article_id", "article_id"),
        Index("ix_article_relevance_statement_id", "statement_id"),
    )

class Article(Base):
    __tablename__ = "articles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, nullable=True)
    author: Mapped[str] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=True)
    published: Mapped[str] = mapped_column(Text, nullable=True)
    accessed: Mapped[str] = mapped_column(Text, nullable=True)
    
    statements = relationship("Statement", secondary="article_relevance", back_populates="articles")
    segments = relationship("Segment", back_populates="article")

    __table_args__ = (
        Index("ix_articles_id", "id"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "content": "", # dont attach content, temporary
            "type": self.type,
            "author": self.author,
            "source": self.source,
            "published": self.published,
            "accessed": self.accessed,
        }

class EvidenceCheat(Base):
    __tablename__ = "evidence_cheat"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(Integer, ForeignKey("statements.id"))
    title: Mapped[str] = mapped_column(Text, default='Článek')
    content: Mapped[str] = mapped_column(Text)
    relevant_paragraph: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float, default=1.0)

class SegmentRelevance(Base):
    __tablename__ = "segment_relevance"
    segment_id: Mapped[int] = mapped_column(Integer, ForeignKey("segments.id", ondelete="CASCADE"), primary_key=True)
    statement_id: Mapped[int] = mapped_column(Integer, ForeignKey("statements.id", ondelete="CASCADE"), primary_key=True)
    relevance: Mapped[float] = mapped_column(Float, default=1.0)

class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(Integer, ForeignKey("articles.id"))
    text: Mapped[str] = mapped_column(Text)
    
    article = relationship("Article", back_populates="segments")
    statements = relationship("Statement", secondary="segment_relevance", back_populates="segments")

    __table_args__ = (
        Index("ix_segments_article_id", "article_id"),
    )

    def to_dict(self, include_relationships: bool = False) -> dict:
        data = {
            "id": self.id,
            "article_id": self.article_id,
            "text": self.text,
        }

        if include_relationships:
            data["article"] = self.article.to_dict() if self.article else None

        return data


class Statement(Base):
    __tablename__ = "statements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(Text)
    date: Mapped[str] = mapped_column(Text)
    party: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    explanation_brief: Mapped[str] = mapped_column(Text)
    origin: Mapped[str] = mapped_column(Text)
    
    articles = relationship("Article", secondary="article_relevance", back_populates="statements")
    segments = relationship("Segment", secondary="segment_relevance", back_populates="statements")
    tags = relationship("Tag", back_populates="statement")

    __table_args__ = (
            Index("ix_statements_id", "id"),
    )

    def __str__(self) -> str:
        return f"{self.statement} - {self.author}, {self.date}"

class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(Integer, ForeignKey("statements.id"))
    tag: Mapped[str] = mapped_column(Text)
    
    statement = relationship("Statement", back_populates="tags")

# Statement filters (used in `.in_()` lookups and joins)

# # Common article lookup
# Index("ix_articles_id", Article.id)
#
# # Segment fast access by article
# Index("ix_segments_article_id", Segment.article_id)
# Index("ix_segments_id", Segment.id)
#
# # SegmentRelevance: join segment → statement and fast lookup
# Index("ix_segment_relevance_segment_id", SegmentRelevance.segment_id)
# Index("ix_segment_relevance_statement_id", SegmentRelevance.statement_id)
#
# # ArticleRelevance: join article → statement and fast lookup
# Index("ix_article_relevance_article_id", ArticleRelevance.article_id)
# Index("ix_article_relevance_statement_id", ArticleRelevance.statement_id)
#
# # EvidenceCheat fast lookup by statement
# Index("ix_evidence_cheat_statement_id", EvidenceCheat.statement_id)
#
# # Tag lookup for a statement
# Index("ix_tags_statement_id", Tag.statement_id)
#
