import os
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

project_path = Path(__file__).parent.parent


def instantiate_model():
    """Задаем параметры модели для обучения

    Returns:
        CatBoostClassifier: Инициированная ML модель
    """
    model = CatBoostClassifier(
        auto_class_weights="Balanced",
        loss_function="MultiClass",
        use_best_model=True,
        eval_metric="MultiClass",
        early_stopping_rounds=50,
    )
    return model


def train():
    """
    Запускаем обучении стратегии и сохраняем обученную модель
    """
    os.makedirs(project_path.as_posix() + "/models", exist_ok=True)

    # считываем обучающие данные
    X = pd.read_parquet(project_path.as_posix() + "/data/processed/X_train.parquet")
    y = pd.read_parquet(project_path.as_posix() + "/data/processed/y_train.parquet")

    # в этом моменте уместно прописать более хитрую схему валидации
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, shuffle=False, test_size=0.3
    )

    model = instantiate_model()

    # Обучаем модель с оптимальным подбором числа деревьев исходя из качества на валидации
    model.fit(y=y_train, X=X_train, eval_set=(X_val, y_val))

    # сохраняем обученную модель
    joblib.dump(model, project_path.as_posix() + "/models/model.joblib")


if __name__ == "__main__":
    train()
