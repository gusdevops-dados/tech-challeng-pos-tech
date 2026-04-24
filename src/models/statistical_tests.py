"""Testes estatísticos usados na validação dos fatores críticos."""

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def chi_square_delay_vs_detractor(df: pd.DataFrame) -> dict:
    """Teste qui-quadrado: atraso na entrega x ser detrator."""
    table = pd.crosstab(df["had_delay"], df["is_detrator"])
    chi2, p_value, dof, expected = chi2_contingency(table)

    return {
        "test": "Qui-quadrado: atraso x detrator",
        "statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "significant_5pct": p_value < 0.05,
        "table": table,
        "expected": expected,
    }


def mann_whitney_delay_breakpoint(df: pd.DataFrame) -> dict:
    """Mann-Whitney: clientes com até 1 dia de atraso vs mais de 1 dia."""
    group_before = df[df["delivery_delay_days"] <= 1]["is_detrator"]
    group_after = df[df["delivery_delay_days"] > 1]["is_detrator"]

    statistic, p_value = mannwhitneyu(group_before, group_after, alternative="less")

    return {
        "test": "Mann-Whitney: atraso <= 1 dia vs > 1 dia",
        "statistic": statistic,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
    }


def mann_whitney_complaints(df: pd.DataFrame) -> dict:
    """Mann-Whitney: clientes sem reclamação vs com reclamação."""
    group_without = df[df["complaints_count"] == 0]["is_detrator"]
    group_with = df[df["complaints_count"] > 0]["is_detrator"]

    statistic, p_value = mannwhitneyu(group_without, group_with, alternative="less")

    return {
        "test": "Mann-Whitney: sem reclamação vs com reclamação",
        "statistic": statistic,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
    }


def run_statistical_tests(df: pd.DataFrame) -> list[dict]:
    """Executa todos os testes estatísticos do projeto."""
    return [
        chi_square_delay_vs_detractor(df),
        mann_whitney_delay_breakpoint(df),
        mann_whitney_complaints(df),
    ]
