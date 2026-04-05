#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dft_int import DFTJobSpec
from src.qc.config import GeometryConfig, PipelineConfig, QuantumTaskConfig
from src.qc.pipeline import QCPipeline


def _load_pipeline_config(path: Path) -> PipelineConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = PipelineConfig()
    geometry_cfg = GeometryConfig(**data.get("geometry", {}))
    quantum_cfg = QuantumTaskConfig(**data.get("quantum", {}))
    pipeline = PipelineConfig(
        geometry=geometry_cfg,
        quantum=quantum_cfg,
        work_dir=Path(data.get("work_dir", defaults.work_dir)),
        max_workers=data.get("max_workers", defaults.max_workers),
        poll_interval=data.get("poll_interval", defaults.poll_interval),
        cleanup_workdir=data.get("cleanup_workdir", defaults.cleanup_workdir),
        store_metadata=data.get("store_metadata", defaults.store_metadata),
        allow_fallback=data.get("allow_fallback", defaults.allow_fallback),
        tracked_properties=tuple(data.get("tracked_properties", defaults.tracked_properties)),
    )
    if isinstance(pipeline.work_dir, str):
        pipeline.work_dir = Path(pipeline.work_dir)
    if isinstance(pipeline.quantum.scratch_dir, str):
        pipeline.quantum.scratch_dir = Path(pipeline.quantum.scratch_dir)
    pipeline.quantum.properties = tuple(pipeline.quantum.properties)
    pipeline.tracked_properties = tuple(pipeline.tracked_properties)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-molecule QC smoke test.")
    parser.add_argument("--smiles", required=True, help="SMILES to test with QC pipeline.")
    parser.add_argument("--qc-config", default="configs/qc_pipeline.yaml", help="QC pipeline YAML.")
    parser.add_argument(
        "--properties",
        default=None,
        help="Comma-separated properties override (default: from qc config). Example: HOMO,LUMO,gap,lambda_max,oscillator_strength",
    )
    parser.add_argument(
        "--require-primary",
        action="store_true",
        help="Fail if fallback was used or status != success.",
    )
    parser.add_argument(
        "--require-terminated",
        action="store_true",
        help="Fail if metadata.terminated_normally is not true.",
    )
    args = parser.parse_args()

    pipe_cfg = _load_pipeline_config(Path(args.qc_config))
    pipeline = QCPipeline(pipe_cfg)

    if args.properties:
        props = [p.strip() for p in args.properties.split(",") if p.strip()]
    else:
        props = list(pipe_cfg.quantum.properties)

    level = pipe_cfg.quantum.level_of_theory or f"{pipe_cfg.quantum.method}/{pipe_cfg.quantum.basis}"
    job = DFTJobSpec(
        smiles=args.smiles,
        properties=props,
        charge=pipe_cfg.quantum.charge,
        multiplicity=pipe_cfg.quantum.multiplicity,
        metadata={"engine": pipe_cfg.quantum.engine, "level_of_theory": level},
    )
    res = pipeline.run(job)

    fallback_used = bool((res.metadata or {}).get("fallback_used", False))
    terminated_normally = (res.metadata or {}).get("terminated_normally")

    print("status:", res.status)
    print("fallback_used:", fallback_used)
    print("terminated_normally:", terminated_normally)
    print("properties:", json.dumps(res.properties, ensure_ascii=False))
    print("workdir:", (res.metadata or {}).get("workdir"))
    if res.error_message:
        print("error_message:", res.error_message)

    failed = False
    if args.require_primary and (res.status != "success" or fallback_used):
        failed = True
    if args.require_terminated and (terminated_normally is not True):
        failed = True
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
