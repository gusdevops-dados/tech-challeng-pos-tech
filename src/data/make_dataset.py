"""Funções para carregamento e checagem inicial dos dados."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Carrega o dataset bruto de NPS."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Retorna um resumo simples de qualidade dos dados."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isna().sum().sort_values(ascending=False),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str),
    }
