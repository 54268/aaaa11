from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_simplified_layout() -> None:
    expected_files = [
        "run_oracle.py",
        "run_wisig.py",
        "run_oracle_subdivision.py",
        "run_wisig_subdivision.py",
        "README.md",
    ]
    expected_dirs = [
        "ablations",
        "functions",
        "settings",
        "data",
        "outputs",
        "third_party",
    ]
    removed_dirs = [
        "sei_osr",
        "tools",
        "trainers",
        "eval",
        "scripts",
        "configs",
        "docs",
        "tests",
        "experiments",
    ]

    for name in expected_files:
        assert (ROOT / name).is_file(), name
    for name in expected_dirs:
        assert (ROOT / name).is_dir(), name
    for name in removed_dirs:
        assert not (ROOT / name).exists(), name
    assert not list(ROOT.glob("*.yaml"))


def test_visualization_font_defaults() -> None:
    import matplotlib.pyplot as plt

    from functions.common import visualization  # noqa: F401

    assert plt.rcParams["font.sans-serif"][0] == "SimHei"
    assert plt.rcParams["axes.unicode_minus"] is False


def test_windows_mkl_thread_env_is_set_early() -> None:
    subdivision_method = (ROOT / "functions" / "methods" / "unknown_subdivision.py").read_text(encoding="utf-8")
    for entry_name in ("run_oracle_subdivision.py", "run_wisig_subdivision.py"):
        entry_text = (ROOT / entry_name).read_text(encoding="utf-8")
        assert "OMP_NUM_THREADS" in entry_text, entry_name
        assert entry_text.index("OMP_NUM_THREADS") < entry_text.index("from functions."), entry_name
    assert subdivision_method.index("OMP_NUM_THREADS") < subdivision_method.index("import numpy")
    assert subdivision_method.index("LOKY_MAX_CPU_COUNT") < subdivision_method.index("import numpy")


def test_function_imports() -> None:
    from functions.methods.boundary_mining import mine_boundary_samples
    from functions.pipeline import run_osr_pipeline
    from functions.methods.pseudo_unknown import generate_hybrid_pseudo_unknown
    from functions.subdivision_pipeline import run_unknown_subdivision

    assert callable(mine_boundary_samples)
    assert callable(generate_hybrid_pseudo_unknown)
    assert callable(run_osr_pipeline)
    assert callable(run_unknown_subdivision)


def test_boundary_and_pseudo_smoke() -> None:
    import numpy as np

    from functions.methods.boundary_mining import mine_boundary_samples
    from functions.methods.pseudo_unknown import generate_hybrid_pseudo_unknown

    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [1.0, 1.0],
            [1.2, 1.1],
            [1.4, 1.3],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    prototypes = np.asarray(
        [
            embeddings[labels == 0].mean(axis=0),
            embeddings[labels == 1].mean(axis=0),
        ],
        dtype=np.float32,
    )

    boundary = mine_boundary_samples(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        k=2,
        alpha=0.5,
        top_m=1,
        ordinary_edge_ratio=0.34,
    )
    pseudo = generate_hybrid_pseudo_unknown(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        boundary_result=boundary,
        ordinary_eta=1.0,
        critical_eta=1.0,
        critical_beta=0.7,
        ordinary_variations=1,
        critical_variations=2,
        jitter=0.0,
        seed=0,
    )

    assert boundary["critical_mask"].sum() == 2
    assert pseudo["summary"]["num_total_pseudo"] == len(pseudo["pseudo_embeddings"])
    assert pseudo["pseudo_embeddings"].shape[1] == 2


def test_unknown_subdivision_smoke() -> None:
    import numpy as np

    from functions.methods.unknown_subdivision import fit_feature_preprocessor, run_ofscil_subdivision

    rng = np.random.default_rng(11)
    cluster_a = rng.normal(loc=[1.0, 0.0], scale=0.03, size=(12, 2))
    cluster_b = rng.normal(loc=[0.0, 1.0], scale=0.03, size=(12, 2))
    raw_features = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    known_anchor = np.asarray([[1.0, 1.0]], dtype=np.float32)

    preprocessor = fit_feature_preprocessor(np.vstack([raw_features, known_anchor]), pca_dim=2)
    features = preprocessor.transform(raw_features)
    anchors = preprocessor.transform(known_anchor)
    result = run_ofscil_subdivision(
        features,
        anchors,
        k_min=2,
        k_max=3,
        seed=11,
        auto_sample_size=0,
        assignment_margin=0.0,
    )

    assert result.labels.shape == (24,)
    assert len(result.centers) >= 1
    assert result.k_search_history


if __name__ == "__main__":
    test_simplified_layout()
    test_visualization_font_defaults()
    test_windows_mkl_thread_env_is_set_early()
    test_function_imports()
    test_boundary_and_pseudo_smoke()
    test_unknown_subdivision_smoke()
    print("项目结构检查通过")
