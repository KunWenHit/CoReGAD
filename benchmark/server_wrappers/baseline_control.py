#!/usr/bin/env python3
"""Compatibility front-end for the server workspace's historical scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys


LAUNCHER = "/data1/wk/codes/CoReGAD_RELEASE/coregad/benchmark/run_baseline.py"
PYTHON = "/data1/wk/conda_envs/pfr_gad/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--dataset", required=True)
    run = sub.add_parser("run")
    run.add_argument("--method", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        command = [PYTHON, LAUNCHER, "--list-methods"]
    elif args.command in {"check", "validate-dataset"}:
        command = [PYTHON, LAUNCHER, "--validate"]
    else:
        command = [
            PYTHON, LAUNCHER, "--method", args.method, "--dataset", args.dataset,
            "--seed", str(args.seed), "--device", args.device, "--protocol", "transductive",
        ]
        if args.execute:
            command.append("--execute")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
