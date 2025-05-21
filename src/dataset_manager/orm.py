"""
Helper functions to interact with the database

@author: Hai Phong Nguyen
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
import time
from .models import *
import logging

logger = logging.getLogger(__name__)

def init_db(path="datasets/curated.sqlite", read_only=False) -> Session:
    if read_only:
        engine = create_engine("sqlite:///")
        import sqlite3
        filedb = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

        start = time.time()
        print("Loading database into memory...")
        filedb.backup(engine.raw_connection().connection)
        print(f"Database loaded to memory in {time.time() - start} seconds")
    else:
        engine = create_engine(f"sqlite:///{path}")

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    Base.metadata.create_all(engine)

    return session

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)

    return d

def rows2dict(rows: list):
    return [row2dict(row) for row in rows]
