"""Funções de visualização principais.
Aqui ficam os gráficos mais importantes do notebook.
Você pode ir movendo novos gráficos para cá aos poucos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

COLORS = {
    "promotor": "#2ecc71",
    "neutro": "#f39c12",
    "detrator": "#e74c3c",
    "accent": "#2980b9",
}


def save_or_show(fig, output_path: str | Path | None = None):
    """Salva a figura se output_path for informado; caso contrário, apenas exibe."""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, numeric_columns: list[str], output_path=None):
    corr_matrix = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns)
    mask_values = mask.values
    import numpy as np

    mask_values[:] = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask_values,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 9},
    )
    ax.set_title("Matriz de Correlação — Variáveis Numéricas", fontsize=14, pad=15)
    fig.tight_layout()
    return save_or_show(fig, output_path)


def plot_model_evaluation(
    model, X_test, y_test, y_pred, y_pred_proba, auc, features, output_path=None
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Detrator", "Detrator"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Matriz de Confusão")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, linewidth=2.5, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (AUC=0.5)")
    axes[1].set_xlabel("Taxa de Falsos Positivos")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].set_title("Curva ROC")
    axes[1].legend()

    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fi_labels = [feature.replace("_enc", "").replace("_", " ") for feature in fi.index]
    axes[2].barh(fi_labels, fi.values)
    axes[2].set_xlabel("Importância")
    axes[2].set_title("Importância das Features\n(Random Forest)")

    fig.tight_layout()
    return save_or_show(fig, output_path)
