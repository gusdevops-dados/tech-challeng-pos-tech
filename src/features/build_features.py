"""Criação de variáveis para EDA e modelagem."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    # Logística
    "delivery_delay_days",
    "delivery_time_days",
    "delivery_attempts",
    "freight_value",
    # Pedido
    "order_value",
    "items_quantity",
    "discount_value",
    "payment_installments",
    # Atendimento
    "customer_service_contacts",
    "resolution_time_days",
    "complaints_count",
    # Cliente
    "customer_age",
    "customer_tenure_months",
    "repeat_purchase_30d",
    # Região
    "customer_region",
]

TARGET = "target_detrator"


def classify_nps(score: float) -> str:
    """Classifica o score NPS em Detrator, Neutro ou Promotor."""
    if score <= 6:
        return "Detrator"
    if score <= 8:
        return "Neutro"
    return "Promotor"


def delay_bucket(days: int | float) -> str:
    """Cria faixas de atraso usadas na análise exploratória."""
    if days == 0:
        return "0 dias (no prazo)"
    if days <= 2:
        return "1-2 dias"
    if days <= 5:
        return "3-5 dias"
    return "Mais de 5 dias"


def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona variáveis auxiliares usadas no EDA e nos testes estatísticos."""
    df = df.copy()
    df["nps_categoria"] = df["nps_score"].apply(classify_nps)
    df["faixa_atraso"] = df["delivery_delay_days"].apply(delay_bucket)
    df["complaints_group"] = df["complaints_count"].apply(
        lambda x: str(min(x, 5)) if x < 5 else "5+"
    )
    df["recompra_label"] = df["repeat_purchase_30d"].map(
        {0: "Sem Recompra", 1: "Com Recompra (30d)"}
    )
    df["is_detrator"] = (df["nps_score"] <= 6).astype(int)
    df["had_delay"] = (df["delivery_delay_days"] > 0).astype(int)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo binária: 1 = detrator, 0 = não detrator."""
    df = df.copy()
    df[TARGET] = (df["nps_score"] <= 6).astype(int)
    return df


def prepare_model_data(df: pd.DataFrame):
    """Prepara X, y, lista de features e encoder para modelagem."""
    df = add_target(df)

    encoder = LabelEncoder()
    df["customer_region_enc"] = encoder.fit_transform(df["customer_region"])

    features_encoded = [
        feature if feature != "customer_region" else "customer_region_enc" for feature in FEATURES
    ]

    X = df[features_encoded]
    y = df[TARGET]
    return X, y, features_encoded, encoder
