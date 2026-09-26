"""
scripts/force_recalibration.py

Retrain the DSS models outside the weekly schedule, inspect the result, and
optionally regenerate the prediction history with the new active models.

Modes (--mode):
  report        Read-only.  Print the active model per pipeline (type, R²,
                MAE, training rows, trained_at) and the gate verdict of its
                last evaluation.  Default when no other mode is given.
  recalibrate   Same code path as the Monday 02:00 UTC cron job
                (ml_engine.training.recalibration.run_recalibration): trains
                every model type, keeps the metric gate, activates the best
                passing candidate, and leaves the current active model in
                place when nothing passes.
  rebootstrap   Same code path as first-start bootstrap
                (ml_engine.training.bootstrap._run_bootstrap_for_pipeline):
                trains every model type and activates the best R² candidate
                whether or not it passes the gate.  Use this to replace a
                degenerate active model (e.g. one trained on a handful of
                rows) that would otherwise keep winning the gate comparison
                by default.  If the dataset is insufficient the current
                active model is left untouched.

Either training mode ends with a per-pipeline table of every candidate the
run produced, so the operator can decide, pipeline by pipeline, whether to
follow up with a different mode.

--reset-predictions (with --yes):
  Delete prediction_results and xai_results for the selected pipelines and
  re-run the ml_engine historical backfill so the history is regenerated
  with the now-active models.  Without --yes it only prints the counts that
  would be deleted.  The unique index on prediction_results is keyed on
  (pipeline, sensor_id, input_feature_timestamp) — not model_id — so old
  rows must be removed before new ones can be written for the same hours.

--pipeline NAME [NAME ...] restricts every mode to those pipelines.

Usage (inside the ml_engine image, e.g. on the VM):
    docker compose run --rm ml_engine python scripts/force_recalibration.py --mode report
    docker compose run --rm ml_engine python scripts/force_recalibration.py --mode recalibrate
    docker compose run --rm ml_engine python scripts/force_recalibration.py --mode rebootstrap --pipeline soil
    docker compose run --rm ml_engine python scripts/force_recalibration.py --reset-predictions --yes

`docker compose run` gives the one-off container the service's env, config
mount and model_artifacts volume, so saved artifacts land where the running
ml_engine loads them from.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "shared"))
sys.path.insert(0, str(_repo))

os.environ.setdefault("DSS_MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("DSS_MONGO_DB_NAME", "dss")
os.environ.setdefault("DSS_ENV", "local")
os.environ.setdefault("DSS_LOG_LEVEL", "INFO")
os.environ.setdefault("DSS_LOG_FORMAT", "human")
os.environ.setdefault("DSS_MODEL_ARTIFACT_PATH", str(_repo / "artifacts"))
os.environ.setdefault("DSS_JWT_SECRET_KEY", "bootstrap_placeholder")
os.environ.setdefault("DSS_ADMIN_API_KEY", "bootstrap_placeholder")

MODES = ("report", "recalibrate", "rebootstrap")


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _fmt(value, width: int = 8, digits: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.{digits}f}".rjust(width)


def _fmt_ts(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "-"


def _plain(value) -> str:
    """Enum member → its value; anything else → str()."""
    return str(getattr(value, "value", value))


def _aware(value: datetime) -> datetime:
    """Mongo may hand back naive UTC datetimes; make them comparable."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def select_pipelines(configs: list, wanted: list[str] | None) -> list:
    """
    Filter the enabled pipeline configs by name.  Unknown names are an
    error rather than a silent no-op.  Pure function for unit tests.
    """
    if not wanted:
        return list(configs)
    by_name = {c.pipeline_name: c for c in configs}
    unknown = sorted(set(wanted) - set(by_name))
    if unknown:
        raise SystemExit(
            f"unknown pipeline(s): {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(by_name))}"
        )
    return [by_name[name] for name in wanted if name in by_name]


def format_candidate_table(rows: list[dict]) -> str:
    """
    Render one run's candidates as a fixed-width table.  `rows` come from
    _collect_run_rows; kept separate so the layout can be unit-tested.
    """
    header = (
        f"{'pipeline':<24} {'model_type':<18} {'status':<10} {'R²':>8} {'MAE':>8} "
        f"{'base_MAE':>8} {'n_train':>7} {'gate':<6} reason"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['pipeline']:<24} {r['model_type']:<18} {r['status']:<10} "
            f"{_fmt(r['r2'])} {_fmt(r['mae'])} {_fmt(r['baseline_mae'])} "
            f"{str(r['n_train']):>7} {('pass' if r['passed'] else 'FAIL'):<6} {r['reason'] or ''}"
        )
    return "\n".join(lines)


async def _latest_metrics(db, model_id: str):
    from dss_shared.db.repositories.model_metrics import ModelMetricsRepository

    docs = await ModelMetricsRepository(db).find_all_for_model(model_id)
    return docs[-1] if docs else None


async def _collect_run_rows(db, pipelines: list[str], since: datetime) -> list[dict]:
    """All registry entries trained during this run, joined with their gate verdict."""
    from dss_shared.db.repositories.model_registry import ModelRegistryRepository

    repo = ModelRegistryRepository(db)
    rows: list[dict] = []
    for pipeline in pipelines:
        docs = await repo.list_all(pipeline=pipeline, limit=100)
        for d in sorted(docs, key=lambda d: (d.model_type, d.trained_at)):
            if _aware(d.trained_at) < since:
                continue
            metrics = await _latest_metrics(db, d.model_id)
            rows.append({
                "pipeline": pipeline,
                "model_type": _plain(d.model_type),
                "status": _plain(d.status),
                "r2": d.metrics_summary.r2,
                "mae": d.metrics_summary.mae,
                "baseline_mae": d.metrics_summary.baseline_mae,
                "n_train": d.training_sample_count,
                "passed": metrics.passed_threshold if metrics else False,
                "reason": (metrics.rejection_reason if metrics else "no metrics doc"),
            })
    return rows


async def report(db, pipelines: list[str]) -> None:
    """Print the active model per pipeline and its last evaluation verdict."""
    from services.ml_engine.registry_manager import find_active_model

    header = (
        f"{'pipeline':<24} {'active model_type':<18} {'R²':>8} {'MAE':>8} {'base_MAE':>8} "
        f"{'n_train':>7} {'trained_at':<17} {'gate':<6} model_id"
    )
    print(header)
    print("-" * len(header))
    for pipeline in pipelines:
        active = await find_active_model(db, pipeline)
        if active is None:
            print(f"{pipeline:<24} (no active model)")
            continue
        metrics = await _latest_metrics(db, active.model_id)
        gate = "-" if metrics is None else ("pass" if metrics.passed_threshold else "FAIL")
        print(
            f"{pipeline:<24} {_plain(active.model_type):<18} "
            f"{_fmt(active.metrics_summary.r2)} {_fmt(active.metrics_summary.mae)} "
            f"{_fmt(active.metrics_summary.baseline_mae)} {str(active.training_sample_count):>7} "
            f"{_fmt_ts(active.trained_at):<17} {gate:<6} {active.model_id}"
        )


async def recalibrate(db, pipelines: list[str]) -> None:
    from services.ml_engine.training.recalibration import run_recalibration

    await run_recalibration(db, pipelines=pipelines)


async def rebootstrap(db, configs: list) -> None:
    from services.ml_engine.training.bootstrap import _run_bootstrap_for_pipeline
    from services.ml_engine.training.recalibration import training_window

    start, end = training_window()
    for cfg in configs:
        await _run_bootstrap_for_pipeline(db, cfg, start, end)


async def reset_predictions(db, pipelines: list[str], confirmed: bool) -> None:
    """Delete prediction/XAI rows for the pipelines, then regenerate history."""
    from dss_shared.db.collections import PREDICTIONS, XAI_RESULTS
    from services.ml_engine.prediction.predictor import run_historical_backfill

    query = {"pipeline": {"$in": pipelines}}
    n_pred = await db[PREDICTIONS].count_documents(query)
    n_xai = await db[XAI_RESULTS].count_documents(query)
    print(f"prediction_results matching {pipelines}: {n_pred}")
    print(f"xai_results matching {pipelines}:        {n_xai}")
    if not confirmed:
        print("Nothing deleted. Re-run with --yes to delete these and regenerate the history.")
        return

    r1 = await db[PREDICTIONS].delete_many(query)
    r2 = await db[XAI_RESULTS].delete_many(query)
    print(f"deleted {r1.deleted_count} predictions, {r2.deleted_count} xai results")

    # Only pipelines with no predictions older than 7 days are backfilled, so
    # pipelines outside --pipeline are skipped by the function's own guard.
    await run_historical_backfill(db)
    for pipeline in pipelines:
        n_after = await db[PREDICTIONS].count_documents({"pipeline": pipeline})
        print(f"{pipeline:<24} predictions now: {n_after}")


async def amain(args: argparse.Namespace) -> None:
    from dss_shared.db import bootstrap_db, get_database, probe_mongo
    from dss_shared.logging import setup_logging
    from services.ml_engine.training.recalibration import _ENABLED_PIPELINES

    setup_logging(level="INFO", fmt="human", service="force_recalibration")

    db = get_database()
    await probe_mongo(db)
    await bootstrap_db(db)

    configs = select_pipelines(_ENABLED_PIPELINES, args.pipeline)
    names = [c.pipeline_name for c in configs]
    run_started = _utc_now()

    if args.mode == "recalibrate":
        await recalibrate(db, names)
    elif args.mode == "rebootstrap":
        await rebootstrap(db, configs)

    if args.mode in ("recalibrate", "rebootstrap"):
        print(f"\nCandidates trained in this run ({args.mode}):")
        print(format_candidate_table(await _collect_run_rows(db, names, run_started)))
        print()

    print("Active models:")
    await report(db, names)

    if args.reset_predictions:
        print()
        await reset_predictions(db, names, confirmed=args.yes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="report", help="report (default) | recalibrate | rebootstrap")
    p.add_argument("--pipeline", nargs="+", metavar="NAME", help="restrict to these pipeline names")
    p.add_argument(
        "--reset-predictions", action="store_true",
        help="delete prediction_results + xai_results for the selected pipelines and regenerate history",
    )
    p.add_argument("--yes", action="store_true", help="confirm the deletion done by --reset-predictions")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
