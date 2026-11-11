#!/usr/bin/env python3
"""Run the CPU contract smoke or GPU LoRA SFT on validated CUA-Lite exports."""
import argparse
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--eval-root', type=Path)
    parser.add_argument('--milestones', type=int, nargs='+', default=[0, 25, 50, 75, 100])
    parser.add_argument('--budget-minutes', type=float, default=60)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('QZ_LINUX_REQUIRED')
    args.output.mkdir(parents=True, exist_ok=False)
    from cua_defense.training import cpu_smoke, train
    if args.cpu_smoke:
        print(json.dumps(cpu_smoke(args.data, args.model, args.output), indent=2))
    else:
        train(args.data, args.model, args.output, steps=args.steps, learning_rate=args.learning_rate,
              accumulation=args.accumulation, seed=args.seed, eval_root=args.eval_root,
              milestones=args.milestones, budget_minutes=args.budget_minutes)


if __name__ == '__main__':
    main()
