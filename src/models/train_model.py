"""Treinamento e comparação de modelos."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Divide os dados em treino e teste mantendo a proporção da target."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_candidate_models(random_state: int = 42):
    """Retorna os modelos candidatos usados no notebook."""
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }


def compare_models(X_train, y_train, random_state: int = 42):
    """Compara modelos usando AUC-ROC em validação cruzada."""
    models = get_candidate_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores

    return results


def train_final_model(X_train, y_train, random_state: int = 42):
    """Treina o modelo campeão do notebook: Random Forest."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
