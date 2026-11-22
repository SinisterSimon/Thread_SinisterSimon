"""App settings, read from environment variables (or a local .env file)."""
import os

from dotenv import load_dotenv

load_dotenv()

# Postgres in production, e.g. postgresql+psycopg://user:pass@host/db
# Falls back to a local SQLite file so the app runs with zero setup.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./thread.db")

# Neon / Render give URLs starting with postgres:// — SQLAlchemy needs the driver name.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

APP_NAME = "Thread"
