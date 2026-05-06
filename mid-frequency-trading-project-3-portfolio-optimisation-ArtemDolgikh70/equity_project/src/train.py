import os
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from equity_project.src.utils import CombinatorialPurgedKFold, PurgedKFold
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

project_path = Path(__file__).parent.parent


def instantiate_model(max_iter):
    """Задаем параметры модели для обучения

    Returns:
        CatBoostClassifier: Инициированная ML модель
    """
    model = CatBoostClassifier(
        auto_class_weights="Balanced",
        loss_function="MultiClass",
        use_best_model=False,
        eval_metric="MultiClass",
        early_stopping_rounds=50,
        iterations=max_iter
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

    a = np.array(sorted(list(set(list(map(lambda x: x[0], list(X.index))))))) #it is time indices, which I take to uniform splitting
    CPCV = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=pd.Series(a + pd.Timedelta(85), index=a))
    for i, (train_id, test_id) in enumerate(CPCV.split(pd.DataFrame(a, index=a))):
        X_train, y_train = X.loc[(a[train_id], slice(None)), :], y.loc[(a[train_id], slice(None)), :]
        X_test, y_test = X.loc[(a[test_id], slice(None)), :], y.loc[(a[test_id], slice(None)), :]
        X_test.to_parquet(project_path.as_posix() + f"/data/processed/X_test_{i}.parquet")
        b = np.array(sorted(list(set(list(map(lambda x: x[0], list(X_train.index))))))) #time indices for train data
        PCV = PurgedKFold(n_splits=2, samples_info_sets=pd.Series(b + pd.Timedelta(85), index=b))
        metrics_min = 10000000000
        #hyperparameters tuning; only two because it is quite a long process
        for max_iter in [100, 2000]:
            metrics = 0
            k = 0
            #Purged CV
            for j, (train_id_j, test_id_j) in enumerate(PCV.split(pd.DataFrame(b, index=b))):
                X_train_j, y_train_j = X_train.loc[(b[train_id_j], slice(None)), :], y_train.loc[(b[train_id_j], slice(None)), :]
                X_test_j, y_test_j = X_train.loc[(b[test_id_j], slice(None)), :], y_train.loc[(b[test_id_j], slice(None)), :]
                model = instantiate_model(max_iter)
                model.fit(X_train_j, y_train_j)
                metrics += model.eval_metrics(Pool(X_test_j, y_test_j), metrics="MultiClass")["MultiClass"][-1]
                k += 1
            metrics /= k
            if metrics < metrics_min:
                metrics_min = metrics
                max_iter_optimum = max_iter
        model = instantiate_model(max_iter_optimum)
        model.fit(X_train, y_train) #train on the whole train data
        # сохраняем обученную модель
        joblib.dump(model, project_path.as_posix() + f"/models/model_{i}.joblib")


if __name__ == "__main__":
    train()
