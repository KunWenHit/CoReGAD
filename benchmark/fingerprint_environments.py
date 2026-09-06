"""Read-only inventory of server baseline environments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/data1/wk/conda_envs")
PACKAGES = {
    "torch": "torch",
    "dgl": "dgl",
    "torch_geometric": "torch-geometric",
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "networkx": "networkx",
    "pandas": "pandas",
    "xgboost": "xgboost",
    "pyod": "pyod",
    "ogb": "ogb",
    "catboost": "catboost",
    "wandb": "wandb",
}


def _probe(python: Path) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,platform\n"
        f"names={PACKAGES!r}\n"
        "out={'python':platform.python_version(),'packages':{}}\n"
        "for key,name in names.items():\n"
        "  try: out['packages'][key]=m.version(name)\n"
        "  except m.PackageNotFoundError: out['packages'][key]=None\n"
        "try:\n"
        " import torch; out['torch_cuda_runtime']=torch.version.cuda; out['torch_cuda_available']=torch.cuda.is_available()\n"
        "except Exception as exc: out['torch_probe_error']=type(exc).__name__+':'+str(exc)\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, check=False, timeout=60)
    if result.returncode:
        return {"status": "BROKEN", "returncode": result.returncode, "stderr_tail": result.stderr.splitlines()[-20:]}
    return {"status": "USABLE", **json.loads(result.stdout)}


def inventory(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
        python = directory / "bin" / "python"
        row: dict[str, Any] = {
            "name": directory.name,
            "path": str(directory.resolve()),
            "python_executable": str(python.resolve()) if python.is_file() else None,
            "protected": directory.resolve() == (root / "pfr_gad").resolve(),
        }
        if python.is_file():
            row.update(_probe(python))
        else:
            row.update({"status": "INCOMPLETE_NO_PYTHON"})
        rows.append(row)
    historical = []
    for python in (
        Path("/data1/wk/Codes/hpp_pyenv_0b3a324/bin/python"),
        Path("/data1/wk/Codes/RHO_HNM_WORKSPACE_S5C_subsample_20260720/.venv/bin/python"),
    ):
        if python.is_file():
            historical.append({"path": str(python.parent.parent), "python_executable": str(python), **_probe(python)})
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scan_root": str(root),
        "policy": "pfr_gad is protected/read-only; incomplete and historical environments are never dispatched",
        "environments": rows,
        "historical_non_dispatch_environments": historical,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inventory(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
