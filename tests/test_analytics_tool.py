from src.tools.analytics_tool import (
    compare_status_groups,
    find_low_quality_cases,
    quality_correlations,
    summarize_operations,
)


def test_summary():
    result = summarize_operations()

    assert len(result) == 1
    assert result.loc[0, "n_rows"] == 1000
    assert 0 <= result.loc[0, "avg_quality"] <= 100


def test_status_comparison():
    result = compare_status_groups()

    statuses = set(result["status"])

    assert "normal" in statuses
    assert "attention" in statuses


def test_low_quality_cases():
    result = find_low_quality_cases(limit=5)

    assert len(result) == 5
    assert result["quality_score"].is_monotonic_increasing


def test_correlations():
    result = quality_correlations()

    assert len(result) == 1
    assert "downtime_correlation" in result.columns