import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database URL - using environment variable for flexibility
# postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE
# CREATE DATABASE copilot;
# CREATE USER copilot_app WITH PASSWORD 'strong_password';
# GRANT CONNECT ON DATABASE copilot TO copilot_app;
# GRANT USAGE ON SCHEMA public TO copilot_app; ~ Can access schema objects
# GRANT CREATE ON SCHEMA public TO copilot_app; ~ Can create tables/indexes/functions
# GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO copilot_app;
# Use(an example): "postgresql://copilot_app:strong_password@localhost/copilot"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/copilot")

# Create database engine
'''
echo = True
This tells SQLAlchemy to print all generated SQL queries to the console/logs.
'''
engine = create_engine(DATABASE_URL, echo=True)

# Create session factory
'''
With autoflush=True, SQLAlchemy may automatically push pending updates before executing the query.
bind=engine ~ This attaches the session factory to the database engine. Sessions created by SessionLocal() will use this engine/DB connection
'''
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
