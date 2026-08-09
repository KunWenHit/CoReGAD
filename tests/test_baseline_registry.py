import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    "Amazon", "Weibo", "YelpChi", "Tolokers", "T-Finance",
    "Elliptic", "T-Social", "DGraph-Fin",
]
EXPECTED = {
    "DOMINANT", "AnomalyDAE", "OCGNN", "AEGIS", "GAAN", "TAM",
    "GAD-NR", "ADA-GAD", "GGAD", "RHO", "GraphNC", "BMP",
    "Structure-aware PU-GNN", "BWGNN", "GHRN", "GADBench / XGBGraph",
    "ConsisGAD", "SpaceGNN", "DSGAD", "APF", "SAGAD", "HSMAD",
}


def _registry():
    return json.loads((ROOT / "benchmark" / "baseline_registry.yaml").read_text(encoding="utf-8"))


def test_active_registry_is_exactly_the_frozen_22():
    registry = _registry()
    rows = registry["baselines"]
    names = [row["method"] for row in rows]
    assert registry["active_count"] == len(rows) == 22
    assert len(set(names)) == 22
    assert set(names) == EXPECTED
    assert not {"PAGE", "TAQ", "TAQ-GAD"} & set(names)


def test_registry_has_complete_protocol_provenance_and_launcher_fields():
    rows = _registry()["baselines"]
    required = {
        "method", "method_id", "main_protocol", "native_training_preserved",
        "ground_truth_anomaly_labels_used_for_training", "source_kind",
        "source_commit", "adapter_status", "reproduction_status",
        "datasets_supported", "strict_oof_available", "fair_ranking",
        "source_url", "license", "environment_id", "adapter", "execute_enabled",
    }
    assert all(required <= row.keys() for row in rows)
    assert all(row["main_protocol"] == "STANDARD_TRANSDUCTIVE_NORMAL_ONLY" for row in rows)
    assert all(row["source_kind"] in {"AUTHOR_OFFICIAL", "OFFICIAL_BENCHMARK_REIMPLEMENTATION", "PAPER_DERIVED"} for row in rows)
    assert all(row["datasets_supported"] == DATASETS for row in rows)
    assert all(row["adapter"] == "benchmark/adapters/transductive_contract.py" for row in rows)
    assert (ROOT / "benchmark" / "run_baseline.py").is_file()


def test_primary_and_supervised_ranking_are_not_conflated():
    rows = _registry()["baselines"]
    primary = [row for row in rows if row["protocol_class"] == "PRIMARY_TRANSDUCTIVE_NORMAL_ONLY"]
    supervised = [row for row in rows if row["protocol_class"] == "EXTERNAL_SUPERVISED_REFERENCE"]
    pu = [row for row in rows if row["protocol_class"] == "PU_AUXILIARY"]
    assert len(primary) == 12 and len(supervised) == 9 and len(pu) == 1
    assert all(not row["ground_truth_anomaly_labels_used_for_training"] and row["fair_ranking"] for row in primary)
    assert all(row["ground_truth_anomaly_labels_used_for_training"] and not row["fair_ranking"] for row in supervised)
    assert pu[0]["method"] == "Structure-aware PU-GNN" and not pu[0]["fair_ranking"]


def test_aegis_and_gaan_provenance_is_benchmark_reimplementation():
    by_name = {row["method"]: row for row in _registry()["baselines"]}
    assert by_name["AEGIS"]["source_kind"] == "OFFICIAL_BENCHMARK_REIMPLEMENTATION"
    assert by_name["GAAN"]["source_kind"] == "OFFICIAL_BENCHMARK_REIMPLEMENTATION"
