from types import SimpleNamespace


def test_detects_data_heavy_tests_by_fixture_name():
    from conftest import _is_data_heavy_item

    item = SimpleNamespace(name="test_any", fixturenames=["features_es"])

    assert _is_data_heavy_item(item) is True


def test_detects_data_heavy_tests_by_test_name():
    from conftest import _is_data_heavy_item

    item = SimpleNamespace(name="test_output_parquets_exist", fixturenames=[])

    assert _is_data_heavy_item(item) is True


def test_leaves_synthetic_tests_unmarked():
    from conftest import _is_data_heavy_item

    item = SimpleNamespace(name="test_ols_pi_coverage_reasonable", fixturenames=[])

    assert _is_data_heavy_item(item) is False
