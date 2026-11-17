"""Run Alembic migrations programmatically (API startup, tests, deploy).

    python -m neuroclass.db.migrate            # upgrade to head
    python -m neuroclass.db.migrate --sql      # print the SQL instead
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config

from ..settings import get_settings

MIGRATIONS = Path(__file__).with_name("migrations")


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str | None = None, sql: bool = False) -> None:
    command.upgrade(alembic_config(url or get_settings().resolved_database_url), "head", sql=sql)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", action="store_true")
    args = parser.parse_args()
    upgrade(sql=args.sql)


if __name__ == "__main__":
    main()
