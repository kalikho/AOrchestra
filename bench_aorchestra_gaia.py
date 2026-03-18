"""
GAIA Benchmark with aorchestra (MainAgent + SubAgent)

Usage:
    python bench_aorchestra_gaia.py --config config/benchmarks/aorchestra_gaia.yaml
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.engine.logs import logger
from benchmark.aorchestra_bench_gaia import GAIAOrchestraBenchmark
from benchmark.gaia.tools import (
    GoogleSearchAction,
    ExecuteCodeAction,
    ExtractUrlContentAction,
    ImageAnalysisAction,
    ParseAudioAction,
)
from aorchestra.config import GAIAOrchestraConfig
from aorchestra.runners.gaia_runner import GAIARunner


DEFAULT_CONFIG_PATH = ROOT / "config/benchmarks/aorchestra_gaia.yaml"


async def main():
    """Run GAIA benchmark with aorchestra."""
    parser = argparse.ArgumentParser(description="Run GAIA benchmark using aorchestra.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--max_concurrency", type=int, default=None, help="Override max_concurrency.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs.")
    parser.add_argument("--skip_completed", type=str, default=None, help="Path to existing CSV to skip completed tasks.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GAIA Benchmark with aorchestra")
    logger.info("=" * 60)

    # Load configuration
    cfg = GAIAOrchestraConfig.load(args.config)

    # Check dataset
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp

    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Attachments: {cfg.attachments_dir}")
    logger.info(f"Level filter: {cfg.level_filter}")

    # Create GAIA tools
    gaia_tools = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
    ]
    logger.info(f"Loaded {len(gaia_tools)} GAIA tools: {[t.name for t in gaia_tools]}")

    # Create benchmark (need to adapt to GAIAOrchestraBenchmark interface)
    from benchmark.bench_gaia import GAIAConfig
    gaia_cfg = GAIAConfig(
        dataset_path=cfg.dataset_path,
        attachments_dir=cfg.attachments_dir,
        max_steps=cfg.max_steps,
        level_filter=cfg.level_filter,
        max_tasks=cfg.max_tasks,
        result_folder=cfg.result_folder,
        trajectory_folder=cfg.trajectory_folder,
    )
    benchmark = GAIAOrchestraBenchmark(gaia_cfg, tools=gaia_tools)
    levels = benchmark.list_levels()

    if not levels:
        logger.error("No tasks found in dataset!")
        return 1

    # Filter by task IDs
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        levels = [l for l in levels if (l.get("task_id") or l.get("id")) in task_ids]
        logger.info(f"Filtered to {len(levels)} task(s)")
    elif cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[:cfg.max_tasks]
        logger.info(f"Limited to {len(levels)} task(s)")

    # Skip completed
    if args.skip_completed:
        skip_csv_path = Path(args.skip_completed)
        if skip_csv_path.exists():
            completed_task_ids = set()
            with open(skip_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row.get("task_id") or row.get("id")
                    if task_id:
                        completed_task_ids.add(task_id)
            
            original_count = len(levels)
            levels = [l for l in levels if (l.get("task_id") or l.get("id")) not in completed_task_ids]
            skipped_count = original_count - len(levels)
            logger.info(f"Skipped {skipped_count} completed task(s)")

    logger.info(f"Running {len(levels)} task(s)")

    # Prepare output paths
    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.result_folder / f"gaia_aorchestra_{timestamp}.csv"

    # Create runner
    runner = GAIARunner(
        benchmark=benchmark,
        main_model=cfg.main_model,
        sub_models=cfg.sub_models,
        max_attempts=cfg.max_attempts,
        gaia_tools=gaia_tools,
    )

    logger.info(f"[GAIA] main_model={cfg.main_model}, sub_models={cfg.sub_models}")
    logger.info(f"Max concurrency: {args.max_concurrency or cfg.max_concurrency}")
    logger.info(f"Results: {csv_path}")

    # Run benchmark
    results = await runner.run_levels(
        levels=levels,
        max_concurrency=args.max_concurrency or cfg.max_concurrency,
        csv_path=csv_path,
        trajectory_folder=cfg.trajectory_folder,
        timestamp=timestamp,
    )

    # Summary
    total = len(results)
    success_count = sum(1 for r in results.values() if r.get("success"))
    total_reward = sum(float(r.get("reward", 0) or 0) for r in results.values())

    logger.info("\n" + "=" * 60)
    logger.info("GAIA Benchmark Summary:")
    logger.info(f"  Total tasks: {total}")
    logger.info(f"  Successful: {success_count}/{total}")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Results: {csv_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
