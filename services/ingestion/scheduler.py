"""
APScheduler setup for the ingestion service.

Reads the ingestion.sources block from config.yaml and registers one
IntervalTrigger job per (source, variable) pair.  Each job runs the full
inline pipeline: fetch → checkpoint filter → preprocess → feature-engineer
→ persist → advance checkpoint.

Source → pipeline mapping:
  wings_water → pipeline="water"   (water quality variables)
  wings_soil  → pipeline="soil"    (soil variables)
  uowm_met    → met variables produce pipeline="met_water" + pipeline="met_soil"
                per the normalizer, but the job key uses the primary pipeline
                ("met_water") and the normalizer handles the duplication.

config.yaml ingestion.sources structure:
  sources:
    - name: wings_water
      variables:
        - name: ph
          poll_interval_seconds: 900
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from motor.motor_asyncio import AsyncIOMotorDatabase

from dss_shared.config import get_raw_yaml
from dss_shared.logging import get_logger
from services.ingestion.clients.registry import get_client
from services.ingestion.jobs.fetch_job import make_fetch_job

log = get_logger(__name__)

# Maps source name from config.yaml → primary pipeline string for checkpoint key.
# Met variables produce readings for two pipelines internally (§E.3),
# but the job uses one checkpoint key per variable.
_SOURCE_PIPELINE: dict[str, str] = {
    "wings_water": "water",
    "wings_soil":  "soil",
    "uowm_met":    "met_water",
}


@dataclass
class _VariableCfg:
    name: str
    poll_interval_seconds: int


@dataclass
class _SourceCfg:
    name: str
    variables: list[_VariableCfg]


def _load_ingestion_sources() -> list[_SourceCfg]:
    """
    Parse the ingestion.sources block from config.yaml.

    Returns an empty list if the block is missing or malformed, so the
    service can still start (with no jobs registered, not a fatal error).
    """
    raw = get_raw_yaml()
    ingestion_block = raw.get("ingestion", {})
    sources_raw = ingestion_block.get("sources", [])
    if not isinstance(sources_raw, list):
        log.warning("ingestion_sources_config_invalid", type=type(sources_raw).__name__)
        return []

    sources: list[_SourceCfg] = []
    for src in sources_raw:
        if not isinstance(src, dict):
            continue
        src_name = src.get("name")
        if not src_name:
            continue
        variables: list[_VariableCfg] = []
        for var in src.get("variables", []):
            var_name = var.get("name")
            interval = var.get("poll_interval_seconds", 900)
            if var_name:
                variables.append(_VariableCfg(name=var_name, poll_interval_seconds=int(interval)))
        sources.append(_SourceCfg(name=src_name, variables=variables))

    return sources


def build_ingestion_scheduler(db: AsyncIOMotorDatabase) -> AsyncIOScheduler:
    """
    Create and configure the ingestion AsyncIOScheduler.

    Returns a scheduler with all jobs registered but not yet started.
    Caller is responsible for calling scheduler.start().
    """
    scheduler = AsyncIOScheduler(timezone="UTC")
    sources = _load_ingestion_sources()

    if not sources:
        log.warning("ingestion_no_sources_configured")

    job_count = 0
    for src_cfg in sources:
        pipeline = _SOURCE_PIPELINE.get(src_cfg.name)
        if pipeline is None:
            log.warning(
                "ingestion_unknown_source",
                source=src_cfg.name,
                hint="Add entry to _SOURCE_PIPELINE in scheduler.py",
            )
            continue

        try:
            client = get_client(src_cfg.name)
        except KeyError:
            log.warning(
                "ingestion_client_not_found",
                source=src_cfg.name,
                hint="Register the client in clients/registry.py",
            )
            continue

        for var_cfg in src_cfg.variables:
            job_fn = make_fetch_job(
                client=client,
                variable_name=var_cfg.name,
                pipeline=pipeline,
                db=db,
            )
            job_id = f"fetch_{src_cfg.name}_{var_cfg.name}"
            scheduler.add_job(
                job_fn,
                trigger="interval",
                seconds=var_cfg.poll_interval_seconds,
                next_run_time=datetime.now(tz=timezone.utc),
                id=job_id,
                name=job_id,
                max_instances=1,    # prevent concurrent runs for same variable
                coalesce=True,      # if job falls behind, run once not many times
                replace_existing=True,
            )
            log.info(
                "ingestion_job_registered",
                source=src_cfg.name,
                variable=var_cfg.name,
                pipeline=pipeline,
                interval_seconds=var_cfg.poll_interval_seconds,
            )
            job_count += 1

    log.info("ingestion_scheduler_built", job_count=job_count)
    return scheduler
