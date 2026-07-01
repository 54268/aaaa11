from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _runner_literal(filename: str, name: str):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {filename}")


def _runner_constant(filename: str, name: str) -> float | int:
    value = _runner_literal(filename, name)
    assert isinstance(value, (float, int))
    return value


def test_full_subdivision_filtering_is_not_overly_aggressive() -> None:
    configs = {
        "oracle": {
            "direct_confidence_quantile": _runner_constant("run_oracle.py", "SUBDIVISION_DIRECT_CONFIDENCE_QUANTILE"),
            "direct_min_cluster_size": _runner_constant("run_oracle.py", "SUBDIVISION_DIRECT_MIN_CLUSTER_SIZE"),
        },
        "wisig": {
            "direct_confidence_quantile": _runner_constant("run_wisig.py", "SUBDIVISION_DIRECT_CONFIDENCE_QUANTILE"),
            "direct_min_cluster_size": _runner_constant("run_wisig.py", "SUBDIVISION_DIRECT_MIN_CLUSTER_SIZE"),
        },
    }

    assert configs["oracle"]["direct_confidence_quantile"] <= 0.05
    assert configs["wisig"]["direct_confidence_quantile"] <= 0.05
    assert 0 < configs["oracle"]["direct_min_cluster_size"] <= 400
    assert 0 < configs["wisig"]["direct_min_cluster_size"] <= 100


def test_main_subdivision_runners_do_not_prespecify_unknown_class_count() -> None:
    for filename in ["run_oracle.py", "run_wisig.py"]:
        assert _runner_constant(filename, "SUBDIVISION_K_MIN") == 2
        assert _runner_constant(filename, "SUBDIVISION_K_MAX") >= 20
        assert _runner_literal(filename, "SUBDIVISION_TARGET_NUM_CLUSTERS") is None
        assert _runner_constant(filename, "SUBDIVISION_TARGET_K_STRENGTH") == 0.0
        assert _runner_literal(filename, "SUBDIVISION_OVERCLUSTER_CANDIDATES") == [0]
        assert _runner_literal(filename, "SUBDIVISION_M_SELECTION_MODE") == "unsupervised"
