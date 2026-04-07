"""
main.py — CLI runner for StartupOpsEnv.

Usage
-----
    python main.py                   # default config
    python main.py --seed 123        # override seed
    python main.py --steps 100       # override max_steps
    python main.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agents.baseline import BaselineAgent
from env.core import StartupOpsEnv
from env.grader import grade_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a StartupOpsEnv episode from the command line."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the RNG seed in config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps in config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step logs.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name to run (investor_pressure, vendor_delay, customer_churn, hiring_crunch).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.steps is not None:
        config["max_steps"] = args.steps
    if args.scenario is not None:
        config["scenario"] = args.scenario
        config["mode"] = "auto"

    env = StartupOpsEnv(config)
    agent = BaselineAgent()

    print(f"\n{'='*60}")
    print(f"  StartupOpsEnv - Baseline Agent")
    print(f"  seed={config['seed']}  max_steps={config['max_steps']}")
    print(f"{'='*60}\n")

    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if args.verbose:
            step = env.time_step
            print(
                f"  Step {step:>3} | action={action['type']:<20} "
                f"| reward={reward:+.4f} | budget={obs.budget:>12,.2f} "
                f"| efficiency={obs.satisfaction:.2%}"
            )

    # ------------------------------------------------------------------
    # Grade & report
    # ------------------------------------------------------------------
    totals = env.get_totals()
    grades = grade_episode(
        logs=env.logs,
        total_emails_created=totals["total_emails_created"],
        total_tasks_created=totals["total_tasks_created"],
        total_negotiations_created=totals["total_negotiations_created"],
        state=env.state,
    )

    print(f"\n{'='*60}")
    print(grades["summary"])
    print(f"{'='*60}\n")

    if args.verbose:
        print("\nFull episode logs (JSON):")
        print(json.dumps(env.logs, indent=2))


if __name__ == "__main__":
    main()
