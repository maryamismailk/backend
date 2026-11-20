from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database connection URL
DATABASE_URL = "mysql+pymysql://root:@localhost/agribot_db"

# connect python to mysql database using sqlalchemy
engine = create_engine(DATABASE_URL)

# Create database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for database models
Base = declarative_base()