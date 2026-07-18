"""
GAIA Benchmark Baseline: Single ReAct agent (no orchestration).

Usage:
    python bench_baseline_gaia.py --config config/benchmarks/gaia.yaml
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.agent.react_agent import ReAcTAgent
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from benchmark.bench_gaia import GAIAConfig, GAIABenchmark, GAIARunner
from benchmark.gaia.tools import (
    GoogleSearchAction,
    ExecuteCodeAction,
    ExtractUrlContentAction,
    ImageAnalysisAction,
    ParseAudioAction,
)


async def main():
    parser = argparse.ArgumentParser(description="Run GAIA baseline (ReAct agent).")
    parser.add_argument("--config", default=str(ROOT / "config/benchmarks/gaia.yaml"))
    parser.add_argument("--max_concurrency", type=int, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GAIA Baseline: ReAct Agent")
    logger.info("=" * 60)

    cfg = GAIAConfig.load(args.config)

    if not cfg.dataset_path.exists():
        logger.error(f"Dataset not found: {cfg.dataset_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    gaia_tools = [
        GoogleSearchAction(),
        ExecuteCodeAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
    ]
    logger.info(f"Loaded {len(gaia_tools)} tools: {[t.name for t in gaia_tools]}")

    benchmark = GAIABenchmark(cfg, tools=gaia_tools)
    levels = benchmark.list_levels()

    if not levels:
        logger.error("No tasks found!")
        return 1

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
        levels = [l for l in levels if (l.get("task_id") or l.get("id")) in task_ids]
    elif cfg.max_tasks and len(levels) > cfg.max_tasks:
        levels = levels[:cfg.max_tasks]

    logger.info(f"Running {len(levels)} tasks with model={args.model}")

    # Create results CSV
    cfg.result_folder.mkdir(parents=True, exist_ok=True)
    cfg.trajectory_folder.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.result_folder / f"gaia_baseline_{timestamp}.csv"

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "task_id", "level", "model", "success", "reward",
        "steps", "total_cost", "timestamp", "error"
    ])
    writer.writeheader()

    llm = create_llm_instance(LLMsConfig.default().get(args.model))

    results = {}
    sem = asyncio.Semaphore(cfg.max_steps or 3)

    async def run_task(level_spec):
        task_id = level_spec.get("task_id") or level_spec.get("id")
        level = level_spec.get("Level")
        question = level_spec.get("Question", "")
        expected = level_spec.get("Final answer", "")

        async with sem:
            env = benchmark.make_env(level_spec, tools=gaia_tools)
            agent = ReAcTAgent(llm=llm)
            agent.reset(env.get_basic_info())

            obs = await env.reset()
            reward = 0.0
            answer = None
            error = None
            steps = 0

            try:
                for step in range(cfg.max_steps):
                    action, resp, _ = await agent.step(obs)
                    obs, reward, done, info = await env.step(action)
                    steps += 1

                    if done:
                        answer = obs.get("submitted_answer")
                        break
            except Exception as e:
                error = str(e)
                logger.error(f"Task {task_id} error: {e}")

            success = reward > 0.5
            results[task_id] = {
                "success": success, "reward": reward,
                "answer": answer, "expected": expected
            }

            writer.writerow({
                "task_id": task_id, "level": level, "model": args.model,
                "success": success, "reward": f"{reward:.4f}", "steps": steps,
                "total_cost": "0.0", "timestamp": timestamp, "error": error,
            })
            csv_file.flush()

            logger.info(f"[Baseline] {task_id}: success={success} reward={reward:.2f}")
            await env.close()

    await asyncio.gather(*[run_task(l) for l in levels])
    csv_file.close()

    total = len(results)
    correct = sum(1 for r in results.values() if r["success"])
    avg_reward = sum(r["reward"] for r in results.values()) / max(total, 1)

    logger.info("\n" + "=" * 60)
    logger.info(f"GAIA Baseline Results:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Tasks: {total}")
    logger.info(f"  Correct: {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    logger.info(f"  Avg reward: {avg_reward:.4f}")
    logger.info(f"  CSV: {csv_path}")
    logger.info("=" * 60)

    # Print final summary for orx log parsing
    print(f"\n=== RESULTS ===")
    print(f"Model: {args.model}")
    print(f"Total tasks: {total}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {100*correct/max(total,1):.1f}%")
    print(f"Avg reward: {avg_reward:.4f}")
    print(f"=== END RESULTS ===")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
