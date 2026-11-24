"""Bring the database schema up to date. Runs at app and worker startup.

Databases created before Alembic existed (Phase 1-2, via create_all) already
have the baseline tables but no record of it. For those, we stamp the
baseline revision first so Alembic only applies the newer migrations instead
of trying to create tables that already exist.

Run by hand with: python -m app.migrate
"""
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
BASELINE = "0001_baseline"


def _config(engine: Engine) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", engine.url.render_as_string(hide_password=False).replace("%", "%%"))
    cfg.attributes["configure_logging"] = False
    return cfg


def upgrade(engine: Engine) -> None:
    cfg = _config(engine)
    with engine.begin() as connection:
        cfg.attributes["connection"] = connection
        tables = set(inspect(connection).get_table_names())
        if "stores" in tables and "alembic_version" not in tables:
            log.info("existing pre-Alembic schema found: stamping %s", BASELINE)
            command.stamp(cfg, BASELINE)
        command.upgrade(cfg, "head")


if __name__ == "__main__":
    from app.db import engine

    logging.basicConfig(level=logging.INFO)
    upgrade(engine)
