"""Pytest safety guards for local market-data tests.

Full 1-minute parquet/pipeline tests are memory-heavy. They are skipped by
default to avoid accidental workstation memory exhaustion. Run explicitly with:

    RUN_DATA_HEAVY=1 python3 -m pytest tests/ -m data_heavy -v
"""

import os

import pytest


DATA_HEAVY_FIXTURES = {
    "es_1min",
    "eth_daily_es",
    "rth_daily_es",
    "rv_features_es",
    "vix",
    "eco",
    "features_es",
}

DATA_HEAVY_TEST_NAMES = {
    "test_output_parquets_exist",
    "test_no_lookahead_in_features",
    "test_build_features_for_returns_eth_and_rth_feature_frames",
}


def _is_data_heavy_item(item) -> bool:
    """Return True when a pytest item will load local parquets or run pipeline."""
    fixture_names = set(getattr(item, "fixturenames", []) or [])
    return bool(fixture_names & DATA_HEAVY_FIXTURES) or item.name in DATA_HEAVY_TEST_NAMES


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "data_heavy: loads local market parquets or runs pipeline scripts; skipped unless RUN_DATA_HEAVY=1",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_DATA_HEAVY") == "1":
        for item in items:
            if _is_data_heavy_item(item):
                item.add_marker(pytest.mark.data_heavy)
        return

    skip_heavy = pytest.mark.skip(reason="data-heavy test skipped; set RUN_DATA_HEAVY=1 to run")
    for item in items:
        if _is_data_heavy_item(item):
            item.add_marker(pytest.mark.data_heavy)
            item.add_marker(skip_heavy)
