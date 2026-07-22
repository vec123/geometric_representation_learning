"""Experiment CLI.

    python scripts/train.py configs/baseline.yaml
    python scripts/train.py configs/baseline.yaml --set training.num_steps=20
    python scripts/train.py configs/ablations/ae_frobenius.yaml --seed 1

Deliberately contains ZERO hyperparameters: everything that describes an
experiment lives in the config file, so an experiment is data, not code.
"""

import argparse
import os
import sys

from config.resolve import (
    resolve_config, config_hash, build_manifest, write_manifest,
)
from src.learning.logger.headless import enable_headless
from src.learning.runner import run_experiment, resolve_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a geometric representation experiment.")
    parser.add_argument("config", nargs="?", help="path to a YAML config (omit for defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY.PATH=VALUE",
                        help="override a config value; repeatable. Unknown keys fail loudly.")
    parser.add_argument("--seed", type=int, help="shorthand for --set seed=<n>")
    parser.add_argument("--device", help="shorthand for --set training.device=<dev>")
    parser.add_argument("--output-dir", help="override where artifacts are written")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve, validate and print the run id, then exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --seed / --device are pure over --set, so there is one override path.
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.device is not None:
        overrides.append(f"training.device={args.device}")

    try:
        cfg, resolved, applied = resolve_config(args.config, overrides)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        # A bad config is a user error, not a crash: fail in the first 50ms with a
        # readable message instead of a traceback (or, worse, at step 300 on a GPU).
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    run_id = config_hash(resolved)

    # Artifacts land under <output_dir>/<run_id> so two configs never overwrite
    # each other's metrics, and re-running the SAME config reuses its directory.
    base_dir = args.output_dir or resolve_path(cfg.output_dir)
    output_dir = os.path.join(base_dir, run_id)

    print(f"run_id: {run_id}  ({cfg.name})")
    if applied:
        print(f"overrides: {applied}")
    if args.dry_run:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    enable_headless(output_dir, remote=None, name="train")
    manifest = build_manifest(resolved, run_id, applied,
                              device=cfg.training.device or "auto",
                              config_path=args.config)
    print(f"manifest: {write_manifest(manifest, output_dir)}")

    run_experiment(cfg, output_dir=output_dir)
    print(f"done. outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())