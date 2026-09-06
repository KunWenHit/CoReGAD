import argparse
import json
from pathlib import Path

from benchmark import direct_reproduction_launcher as launcher


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmark" / "baseline_runtime_registry.json"
GATES = {
    "SOURCE_PASS",
    "ENV_PASS",
    "NATIVE_SANITY_PASS",
    "LABEL_OR_SUPERVISION_AUDIT_PASS",
    "CANONICAL_BRIDGE_PASS",
    "SCORE_CONTRACT_PASS",
    "LAUNCHER_PASS",
}


def _registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_runtime_registry_is_exactly_21_and_excludes_retired_methods():
    registry = _registry()
    methods = registry["methods"]
    names = {row["method"] for row in methods}
    classes = [row["protocol_class"] for row in methods]
    assert registry["active_total"] == len(methods) == 21
    assert registry["required_gates"] == list(launcher.REQUIRED_GATES)
    assert (classes.count("PRIMARY_TRANSDUCTIVE_NORMAL_ONLY"), classes.count("PU_AUXILIARY"), classes.count("EXTERNAL_SUPERVISED_REFERENCE")) == (11, 1, 9)
    assert not names & {"BMP", "PAGE", "TAQ", "TAQ-GAD"}


def test_execute_enabled_requires_all_seven_pass_gates_and_real_command():
    assert sum(bool(row["execute_enabled"]) for row in _registry()["methods"]) == 21
    for row in _registry()["methods"]:
        assert set(row["gates"]) == GATES
        if row["execute_enabled"]:
            assert all(gate["status"] == "PASS" for gate in row["gates"].values())
            assert row["canonical_entrypoint"]
            assert isinstance(row["canonical_command"], list)
            assert row["blocker"] is None
        else:
            assert row["blocker"]["class"] in {
                "ENV_BLOCKED", "SOURCE_BLOCKED", "IMPLEMENTATION_BLOCKED",
                "RUNTIME_BLOCKED", "RESOURCE_BLOCKED",
            }


def test_hsmad_is_frozen_author_official_checkout():
    hsmad = next(row for row in _registry()["methods"] if row["method"] == "HSMAD")
    assert hsmad["source_commit"] == "7810e8e7dfe2143c4e4aa2ba804a0bfdc6543a5a"
    assert hsmad["native_source"] == "baselines/repos/HSMAD"


def test_all_21_methods_dry_run_on_one_canonical_dataset_without_training():
    for row in _registry()["methods"]:
        args = argparse.Namespace(dataset="Weibo", seed=0, device="auto", execute=False)
        plan = launcher._dry_run(launcher._resolve_method(row["method"]), args)
        assert plan["method"] == row["method"]
        assert plan["dataset"] == "Weibo"
        assert plan["output_directory"].endswith(f"/{row['output_name']}/Weibo/seed_0") or plan["output_directory"].endswith(f"\\{row['output_name']}\\Weibo\\seed_0")


def test_server_assets_exist_for_every_enabled_runtime():
    release = Path("/data1/wk/codes/CoReGAD_RELEASE")
    if not release.is_dir():
        return
    for row in _registry()["methods"]:
        source_entrypoint = release / row["native_source"] / row["native_entrypoint"]
        assert source_entrypoint.is_file(), row["method"]
        if row["execute_enabled"]:
            assert Path(row["environment_path"]).is_dir(), row["method"]
            assert Path(row["python_executable"]).is_file(), row["method"]
            assert (release / "coregad" / row["canonical_entrypoint"]).is_file(), row["method"]


def test_unified_output_contract_is_declared_by_launcher():
    source = (ROOT / "benchmark" / "direct_reproduction_launcher.py").read_text(encoding="utf-8")
    for name in ("run_manifest.json", "score.npz", "metrics.json", "stdout.log", "stderr.log", "resource.json", "resource_failure.json"):
        assert name in source
    shell = (ROOT / "scripts" / "run_baseline.sh").read_text(encoding="utf-8")
    assert "direct_reproduction_launcher.py" in shell
    assert "--method" not in shell
