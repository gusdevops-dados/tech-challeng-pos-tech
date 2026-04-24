"""Testes básicos do pipeline de dados e features."""

import pandas as pd

from src.data.make_dataset import data_quality_summary, load_data
from src.features.build_features import (
    add_analysis_features,
    classify_nps,
    delay_bucket,
    prepare_model_data,
)


def test_load_data_reads_csv(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("nps_score,delivery_delay_days\n10,0\n5,3\n")

    df = load_data(file_path)

    assert df.shape == (2, 2)
    assert list(df.columns) == ["nps_score", "delivery_delay_days"]


def test_data_quality_summary_returns_expected_keys():
    df = pd.DataFrame({"a": [1, 2], "b": [None, 3]})

    summary = data_quality_summary(df)

    assert summary["shape"] == (2, 2)
    assert summary["duplicates"] == 0
    assert "missing_values" in summary
    assert "columns" in summary


def test_classify_nps():
    assert classify_nps(3) == "Detrator"
    assert classify_nps(7) == "Neutro"
    assert classify_nps(10) == "Promotor"


def test_delay_bucket():
    assert delay_bucket(0) == "0 dias (no prazo)"
    assert delay_bucket(2) == "1-2 dias"
    assert delay_bucket(5) == "3-5 dias"
    assert delay_bucket(6) == "Mais de 5 dias"


def test_add_analysis_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "nps_score": [10, 5],
            "delivery_delay_days": [0, 4],
            "complaints_count": [0, 6],
            "repeat_purchase_30d": [1, 0],
        }
    )

    result = add_analysis_features(df)

    expected_columns = {
        "nps_categoria",
        "faixa_atraso",
        "complaints_group",
        "recompra_label",
        "is_detrator",
        "had_delay",
    }

    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "is_detrator"] == 0
    assert result.loc[1, "is_detrator"] == 1


def test_prepare_model_data_returns_x_y_features_and_encoder():
    df = pd.DataFrame(
        {
            "nps_score": [10, 5, 8, 2],
            "delivery_delay_days": [0, 4, 1, 7],
            "delivery_time_days": [2, 6, 3, 9],
            "delivery_attempts": [1, 2, 1, 3],
            "freight_value": [10.0, 20.0, 15.0, 30.0],
            "order_value": [100.0, 200.0, 150.0, 300.0],
            "items_quantity": [1, 2, 1, 3],
            "discount_value": [0.0, 5.0, 10.0, 0.0],
            "payment_installments": [1, 2, 3, 4],
            "customer_service_contacts": [0, 2, 1, 3],
            "resolution_time_days": [0, 5, 2, 8],
            "complaints_count": [0, 2, 1, 5],
            "customer_age": [25, 40, 32, 50],
            "customer_tenure_months": [12, 24, 6, 36],
            "repeat_purchase_30d": [1, 0, 1, 0],
            "customer_region": ["SP", "RJ", "SP", "MG"],
        }
    )

    x, y, features, encoder = prepare_model_data(df)

    assert x.shape[0] == 4
    assert len(y) == 4
    assert "customer_region_enc" in features
    assert "customer_region" not in x.columns
    assert hasattr(encoder, "classes_")
