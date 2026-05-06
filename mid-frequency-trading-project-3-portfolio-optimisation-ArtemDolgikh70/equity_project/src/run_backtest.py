import os
from pathlib import Path

import joblib
import pandas as pd
import vectorbt as vbt
import numpy as np
import matplotlib.pyplot as plt

from equity_project.src.utils import load_config, save_dict

project_path = Path(__file__).parent.parent


def generate_weights(preds):
    """Превращаем скоры ML модели в веса бумаг в портфеле

    Args:
        preds (pd.DataFrame): Датафрейм скоров ML модели

    Returns:
        pd.DataFrame: Веса бумаг в портфеле
    """
    preds_unstack = preds.unstack(level=1)

    #it is I do not use
    long_prob_minus_short_prob = preds_unstack[2] - preds_unstack[0]
    signals_rank = long_prob_minus_short_prob.rank(axis=1, ascending=False, pct=False)
    weights = np.array(signals_rank)

    for i in range(len(weights)):
        if i == 0:
            weights[i, :] = 0
        else:
            weights[i, :] = weights[i - 1, :] #set the same weight as in the previous check
        if i % 80 == 5: #every 80 days change weights: if the biggest possibility is to hold then do not change anything, if the biggest possibility is to buy, set 1, if to sell, set 0
            weights[i, (preds_unstack[2].iloc[i, :] > preds_unstack[0].iloc[i, :]) & (preds_unstack[2].iloc[i, :] > preds_unstack[1].iloc[i, :])] = 1
            weights[i, (preds_unstack[0].iloc[i, :] > preds_unstack[1].iloc[i, :]) & (preds_unstack[0].iloc[i, :] > preds_unstack[2].iloc[i, :])] = 0

    # делим ранг на сумму рангов бумаг, таким образом в портфеле дается больше веса тем бумагам, которые имеют бОльшую вероятность роста (пропорционально рангу)
    weights = pd.DataFrame((weights.T / max(weights.sum(axis=1))).T, columns=signals_rank.columns, index=signals_rank.index)
    weights = weights.fillna(0)
    return weights


def run_backtest():
    """
    Запускает бэктест на бэктестовых данных
    Сохраняет:
        - Основные бэктестовые метрики в /artifacts/backtest_metrics.json
        - График PnL стратегии в /artifacts/pnl.png
    """

    os.makedirs(project_path.as_posix() + "/artifacts/plots", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/artifacts/metrics", exist_ok=True)

    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    # считываем бэктестовые данные и ML модель
    X_backtest = pd.read_parquet(
        project_path.as_posix() + "/data/processed/X_backtest.parquet"
    )

    backtest_data = pd.read_parquet(
        project_path.as_posix() + "/data/raw/backtest_data.parquet", engine="pyarrow"
    )

    paths = {}
    #construct paths from CPCV chunks
    for i in range(15):
        X_test = pd.read_parquet(
            project_path.as_posix() + f"/data/processed/X_test_{i}.parquet"
        )
        model = joblib.load(project_path.as_posix() + f"/models/model_{i}.joblib")
        prediction = pd.DataFrame(model.predict_proba(X_test), index=X_test.index)
        if i < 5:
            paths[i] = prediction
        else:
            if i == 5:
                i1 = 3
            if i == 6:
                i1 = 4
            if i == 7:
                i1 = 2
            if i == 8:
                i1 = 1
            if i == 9:
                i1 = 0
            if i == 10:
                i1 = 4
            if i == 11:
                i1 = 2
            if i == 12:
                i1 = 1
            if i == 13:
                i1 = 3
            if i == 14:
                i1 = 0
            paths[i1] = pd.concat([paths[i1], prediction], axis=0)
    
    for i in paths:
        paths[i] = paths[i].sort_index()

    weights_average = 0
    for i in paths:
        close = backtest_data.Close.dropna(axis=1, how="all")
        preds = paths[i].loc[close.index] #choose test paths
        size = generate_weights(preds)
        price = backtest_data.shift(-1).Open[list(set(close.columns) & set(size.columns))]
        close = close[price.columns]
        size = size[price.columns]

        #it is for average path
        if type(weights_average) is int:
            weights_average = 0.2 * size
        else:
            weights_average += 0.2 * size
        # формируем портфель на основе сигналов
        init_cash = cfg["init_cash"]
        fees = cfg["fees"]

        pf = vbt.Portfolio.from_orders(
            close=close,
            price=price,
            size=size,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
            freq="1d",
            init_cash=init_cash,
            fees=fees,
        )

        # сохраняем PnL график
        pf.plot().write_image(project_path.as_posix() + f"/artifacts/plots/pnl_{i}.png")

        # сохраняем метрики бэктеста
        backtest_metrics = pf.stats().to_dict()
        save_dict(
            backtest_metrics,
            project_path.as_posix() + f"/artifacts/metrics/backtest_metrics_{i}.json",
        )

    pf = vbt.Portfolio.from_orders(
            close=close,
            price=price,
            size=weights_average,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
            freq="1d",
            init_cash=init_cash,
            fees=fees,
        )

    # сохраняем PnL график
    pf.plot().write_image(project_path.as_posix() + f"/artifacts/plots/pnl.png")

    # сохраняем метрики бэктеста
    backtest_metrics = pf.stats().to_dict()
    save_dict(
        backtest_metrics,
        project_path.as_posix() + f"/artifacts/metrics/backtest_metrics.json",
    )


if __name__ == "__main__":
    run_backtest()