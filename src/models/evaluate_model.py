"""Avaliação do modelo."""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_classifier(model, X_test, y_test):
    """Retorna métricas principais do classificador."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Não-Detrator", "Detrator"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics, y_pred, y_pred_proba


def feature_importance(model, features: list[str]) -> pd.Series:
    """Retorna a importância das variáveis do Random Forest."""
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
