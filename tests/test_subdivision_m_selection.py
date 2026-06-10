from functions.subdivision_pipeline import _select_minimal_sufficient_m


def test_oracle_selects_m2_when_quality_gain_is_material_without_final_redundancy():
    candidates = [
        {
            "overcluster_extra_clusters": 0,
            "resolved_num_clusters": 5,
            "m_selection_score": -4.306172,
            "m_selection_offline_adjusted_quality": 0.700000,
        },
        {
            "overcluster_extra_clusters": 1,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.798965,
            "m_selection_offline_adjusted_quality": 0.805900,
        },
        {
            "overcluster_extra_clusters": 2,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.757805,
            "m_selection_offline_adjusted_quality": 0.839100,
        },
        {
            "overcluster_extra_clusters": 3,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.726511,
            "m_selection_offline_adjusted_quality": 0.816800,
        },
    ]

    selected = _select_minimal_sufficient_m(
        candidates,
        target_num_clusters=6,
        min_quality_gain=0.01,
    )

    assert selected["overcluster_extra_clusters"] == 2


def test_later_material_gain_is_not_blocked_by_an_early_small_gain():
    candidates = [
        {
            "overcluster_extra_clusters": 0,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.10,
            "m_selection_offline_adjusted_quality": 0.700000,
        },
        {
            "overcluster_extra_clusters": 1,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.11,
            "m_selection_offline_adjusted_quality": 0.704000,
        },
        {
            "overcluster_extra_clusters": 2,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.12,
            "m_selection_offline_adjusted_quality": 0.760000,
        },
        {
            "overcluster_extra_clusters": 3,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.13,
            "m_selection_offline_adjusted_quality": 0.755000,
        },
    ]

    selected = _select_minimal_sufficient_m(
        candidates,
        target_num_clusters=6,
        min_quality_gain=0.01,
    )

    assert selected["overcluster_extra_clusters"] == 2


def test_one_percentage_point_gain_is_not_enough_to_add_structure():
    candidates = [
        {
            "overcluster_extra_clusters": 0,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.10,
            "m_selection_offline_adjusted_quality": 0.700000,
        },
        {
            "overcluster_extra_clusters": 1,
            "resolved_num_clusters": 6,
            "m_selection_score": 0.11,
            "m_selection_offline_adjusted_quality": 0.710000,
        },
    ]

    selected = _select_minimal_sufficient_m(
        candidates,
        target_num_clusters=6,
        min_quality_gain=0.01,
    )

    assert selected["overcluster_extra_clusters"] == 0
