from __future__ import annotations

import numpy as np
import pandas as pd

from carbon_utils import (
    parse_numeric,
    parse_fraction,
    normalize_country,
    simplify_protocol,
    weight_category,
    fit_label_thresholds,
    apply_carbon_labels,
    calculate_lca_bottom_up,
    hybrid_pcf_estimate,
    detect_outliers,
    fmt_num,
    safe_mape,
    median_ape,
)
from scenario_projection import (
    normalize_lifecycle_shares,
    build_custom_2050_factors,
    interpolate_factor,
    uncertainty_rate,
    project_pcf_for_year,
    build_projection_pathway,
    build_assumptions_table,
)


def test_parse_numeric_plain_number():
    assert parse_numeric("12.5") == 12.5


def test_parse_numeric_with_comma():
    assert parse_numeric("1,234.5") == 1234.5


def test_parse_numeric_invalid_is_nan():
    assert np.isnan(parse_numeric("abc"))


def test_parse_fraction_percent():
    assert abs(parse_fraction("40%") - 0.40) < 1e-9


def test_parse_fraction_plain_ratio():
    assert abs(parse_fraction("0.25") - 0.25) < 1e-9


def test_normalize_country_alias_usa():
    assert normalize_country("USA") == "United States Of America"


def test_normalize_country_vietnam_alias():
    assert normalize_country("Viet Nam") == "Vietnam"


def test_simplify_protocol_empty():
    assert simplify_protocol("") == "Unknown"


def test_simplify_protocol_contains_iso():
    assert simplify_protocol("ISO 14067") == "ISO"


def test_weight_category_small():
    assert weight_category(0.2) == "Very light (<1 kg)"


def test_weight_category_heavy():
    assert weight_category(150) == "Heavy (100-1000 kg)"


def test_fit_label_thresholds_train_only():
    df = pd.DataFrame({"pcf_kg_co2e": [1, 2, 3, 4, 5]})
    thresholds = fit_label_thresholds(df)
    assert "q25" in thresholds and "q75" in thresholds
    assert thresholds["q25"] <= thresholds["q75"]


def test_apply_carbon_labels_three_levels():
    df = pd.DataFrame({"pcf_kg_co2e": [1, 5, 10]})
    labels = apply_carbon_labels(df, {"q25": 2, "q75": 8}).tolist()
    assert labels == ["Low", "Medium", "High"]


def test_calculate_lca_bottom_up_total():
    inv = pd.DataFrame([
        {"stage": "Material", "activity_group": "Material", "activity_name": "Steel", "unit": "kg", "amount": 2, "emission_factor": 1.5, "quality": "High"},
        {"stage": "Energy", "activity_group": "Energy", "activity_name": "Electricity", "unit": "kWh", "amount": 4, "emission_factor": 0.5, "quality": "Medium"},
    ])
    out = calculate_lca_bottom_up(inv)
    assert abs(out["total_pcf"] - 5.0) < 1e-9
    assert not out["by_group"].empty


def test_hybrid_pcf_estimate_with_lca():
    assert abs(hybrid_pcf_estimate(10, 20, lca_weight=0.25) - 12.5) < 1e-9


def test_hybrid_pcf_estimate_without_lca():
    assert hybrid_pcf_estimate(10, 0) == 10


def test_detect_outliers_iqr():
    df = pd.DataFrame({"x": [1, 1, 1, 2, 2, 100]})
    out = detect_outliers(df, "x", threshold=1.5)
    assert 100 in out["x"].tolist()


def test_detect_outliers_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert detect_outliers(df, "missing").empty


def test_fmt_num_none():
    assert fmt_num(None) == "-"


def test_safe_mape_zero_safe():
    val = safe_mape([0, 1], [0, 2])
    assert np.isfinite(val)


def test_median_ape_basic():
    val = median_ape([10, 20], [11, 18])
    assert val >= 0


def test_normalize_lifecycle_default_sums_to_one():
    shares = normalize_lifecycle_shares()
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_normalize_lifecycle_zero_fallback():
    shares = normalize_lifecycle_shares(0, 0, 0, 0, 0)
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert shares["upstream"] > 0


def test_build_custom_2050_factors_bounds():
    factors = build_custom_2050_factors(100, 80, 80, 80)
    assert all(0 < v <= 1.10 for v in factors.values())


def test_interpolate_factor_baseline_year():
    assert interpolate_factor(2025, 2025, 2050, 0.5) == 1.0


def test_interpolate_factor_target_year():
    assert abs(interpolate_factor(2050, 2025, 2050, 0.5) - 0.5) < 1e-9


def test_uncertainty_rate_2030():
    assert uncertainty_rate(2030) == 0.15


def test_uncertainty_rate_2045():
    assert uncertainty_rate(2045) == 0.30


def test_project_pcf_for_year_output_keys():
    shares = normalize_lifecycle_shares()
    out = project_pcf_for_year(10, 2030, 2025, "baseline", shares)
    assert out["projected_pcf"] > 0
    assert out["upper"] > out["lower"]


def test_build_projection_pathway_years():
    shares = normalize_lifecycle_shares()
    df = build_projection_pathway(10, 2025, 2030, "net_zero", shares)
    assert df["year"].min() == 2025
    assert df["year"].max() == 2030
    assert len(df) == 6


def test_build_assumptions_table_not_empty():
    table = build_assumptions_table()
    assert not table.empty
    assert "Kịch bản" in table.columns


def test_imbalance_class_weights_and_distribution():
    import numpy as np
    from imbalance_handler import class_distribution, compute_balanced_class_weights
    y = np.array([0] * 10 + [1] * 4 + [2] * 1)
    dist = class_distribution(y)
    weights = compute_balanced_class_weights(y)
    assert dist == {0: 10, 1: 4, 2: 1}
    assert weights[2] > weights[1] > weights[0]


def test_sensitivity_outputs_shape():
    from sensitivity_analysis import one_way_sensitivity, tornado_chart
    df = one_way_sensitivity(100, "Material", 1.9, lifecycle_share=0.4)
    assert not df.empty
    assert "pcf_new" in df.columns
    tornado = tornado_chart(100, output_file="outputs/figures/test_sensitivity_tornado.png")
    assert not tornado.empty
    assert "range" in tornado.columns
