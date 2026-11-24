"""Background worker: picks up queued experiments and runs them.

Run it with `python -m worker.run` (loops forever) or
`python -m worker.run --once` (one pass, handy locally).
"""
import argparse
import logging
import time

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.experiments import run_experiment
from app.llm import get_llm
from app.models import Experiment, ExperimentStatus

log = logging.getLogger("worker")
POLL_SECONDS = 5


def claim_next(session) -> Experiment | None:
    """Take the oldest queued experiment.

    On Postgres, SKIP LOCKED lets several workers run without grabbing the
    same experiment twice. SQLite ignores it, which is fine for one worker.
    """
    stmt = (
        select(Experiment)
        .where(Experiment.status == ExperimentStatus.queued)
        .order_by(Experiment.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    return session.scalars(stmt).first()


def run_once() -> int:
    ran = 0
    with SessionLocal() as session:
        while (exp := claim_next(session)) is not None:
            log.info("running experiment %s (%s)", exp.id, exp.agent_key)
            run_experiment(session, exp, get_llm(shadow=exp.shadow))
            ran += 1
    return ran


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    init_db()
    if args.once:
        log.info("ran %d experiment(s)", run_once())
        return
    while True:
        if run_once() == 0:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
