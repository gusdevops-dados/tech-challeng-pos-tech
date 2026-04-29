"""Pipeline principal do projeto de NPS preditivo.

Execute a partir da raiz do projeto:

    python -m src.pipeline

O arquivo bruto esperado é:
    data/raw/desafio_nps_fase_1.csv
"""

from pathlib import Path

import joblib

from src.data.make_dataset import load_data
from src.features.build_features import add_analysis_features, prepare_model_data
from src.models.evaluate_model import evaluate_classifier, feature_importance
from src.models.statistical_tests import run_statistical_tests
from src.models.train_model import compare_models, split_data, train_final_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desafio_nps_fase_1.csv"
MODEL_PATH = ROOT / "models" / "random_forest_nps.pkl"
REPORTS_PATH = ROOT / "reports"


def main():
    df = load_data(DATA_PATH)
    df = add_analysis_features(df)

    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")

    print("\n=== Testes estatísticos ===")
    for result in run_statistical_tests(df):
        print(
            f"{result['test']}: "
            f"estatística={result['statistic']:.4f}, "
            f"p-valor={result['p_value']:.6f}, "
            f"significativo={result['significant_5pct']}"
        )

    X, y, features, encoder = prepare_model_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n=== Comparação de modelos por AUC-ROC ===")
    comparison = compare_models(X_train, y_train)
    for name, scores in comparison.items():
        print(f"{name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    model = train_final_model(X_train, y_train)
    metrics, _, _ = evaluate_classifier(model, X_test, y_test)

    print("\n=== Modelo final ===")
    print(f"AUC-ROC teste: {metrics['auc_roc']:.4f}")
    print(f"Metricas de classificacao: {metrics['classification_report']}")

    fi = feature_importance(model, features)
    print("\n=== Top 10 variáveis mais importantes ===")
    print(fi.head(10))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "features": features,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
