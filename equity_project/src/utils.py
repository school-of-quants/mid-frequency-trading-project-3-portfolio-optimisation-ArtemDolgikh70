import datetime as dt
import json
from typing import Dict

import pandas as pd
from yaml import safe_load

from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection._split import KFold


def load_config(config_path: str) -> Dict:
    """Загружает yaml конфиг в виде python словаря

    Args:
        config_path (str): Путь до конфига

    Returns:
        Dict: Словарь с параметрами конфига
    """
    with open(config_path) as file:
        config = safe_load(file)
    return config


def save_dict(dict_, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dict_, f, indent=4, default=str)


def applyPtSlOnT1(close, events, ptSl):
    """Tripple barrier method adjusted for Low and High prices

    Args:
        close (pd.Series): Close prices.
        events (pd.DataFrame): A pandas dataframe, with columns:
            - t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
            - trgt: The unit width of the horizontal barriers.
        ptSl (list):
            - ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                If 0, there will not be an upper barrier.
            - ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                If 0, there will not be a lower barrier.

    Returns:
        pd.DataFrame: Timestamps of each barrier touch
    """

    # apply stop loss/profit taking, if it takes place before t1 (end of event)

    out = events[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = df0 / close[loc] - 1  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def three_barrier(close, ptSl=[1, 1], rolling_n=50, scaling_factor=2.0):
    """Labeling based on tripple barrier method and historical volatility

    Args:
        close (pd.DataFrame): Assets close prices
        ptSl (list, optional): applyPtSlOnT1 ptSl parameter. Defaults to [1, 1].
        rolling_n (int, optional): Rolling window to calculate standart deviation. Defaults to 50.
        scaling_factor (float, optional): Multiplier of standart deviation to calculate horizontal barrier size. Defaults to 2.0.

    Returns:
        pd.Series: 3-class labels
    """
    events = pd.DataFrame(
        {
            "t1": close.index + dt.timedelta(days=80),
            "trgt": 0.3,
        },
        index=close.index,
    )
    out = applyPtSlOnT1(close, events, ptSl)
    target = out.apply(
        lambda x: 1 if x.idxmin() == "pt" else -1 if x.idxmin() == "sl" else 0, axis=1
    )

    return target








class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(
        self,
        n_splits: int = 3,
        samples_info_sets: pd.Series = None,
        pct_embargo: float = 0.0,
    ):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
            For example, in the case of three-barrier labeling, samples_info_sets.value could be the minimum date of touching a particular barrier.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """
        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.samples_info_sets = samples_info_sets
        self.pctEmbargo = pct_embargo

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X: (pd.DataFrame) Samples dataset that is to be split.
        :param y: (pd.Series) Sample labels series.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        if (X.index == self.samples_info_sets.index).sum() != len(
            self.samples_info_sets
        ):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)

        # для каждой тестовой выборки считаем прочищенную обучающую, потом выдаем пересечение обучающих
        splits = super().split(X, y, groups)
        for base_train_indices, base_test_indices in splits:
            i = base_test_indices[0]
            t0 = self.samples_info_sets.index[i]  # start of test set
            maxT1Idx = self.samples_info_sets.index.searchsorted(
                self.samples_info_sets[base_test_indices].max()
            )
            train_indices = self.samples_info_sets.index.searchsorted(
                self.samples_info_sets[self.samples_info_sets <= t0].index
            )
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx + mbrg :])
                )
            yield train_indices, base_test_indices


class CombinatorialPurgedKFold(KFold):
    """

    Implements Combinatorial Purged Cross Validation (CPCV).

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        samples_info_sets: pd.Series = None,
        pct_embargo: float = 0.0,
    ):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 6
        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        super(CombinatorialPurgedKFold, self).__init__(
            n_splits, shuffle=False, random_state=None
        )
        self.n_test_splits = n_test_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def _generate_combinatorial_test_ranges(self, splits_indices: dict):
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        test_ids = tuple(combinations(splits_indices, r=self.n_test_splits))
        func = lambda x: tuple((i for i in splits_indices if i not in x))
        train_ids = tuple([func(i) for i in test_ids])
        ids = zip(train_ids, test_ids)
        return ids

    def _fill_backtest_paths(self, indices: list, test_indices: dict, mbrg: int):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param indices: (list) List of indices of the sample
        :param test_indices: (dict of lists) Key: number of split, value: indices of sample with this number.
        """

        train_mask = np.ones(len(indices), dtype=bool)

        for i in test_indices:
            # train_mask[test_indices[i]] = False

            t0 = self.samples_info_sets.index[test_indices[i][0]]  # start of test set
            maxT1Idx = self.samples_info_sets.index.searchsorted(
                self.samples_info_sets[test_indices[i]].max()
            )
            train_indices = self.samples_info_sets.index.searchsorted(
                self.samples_info_sets[self.samples_info_sets <= t0].index
            )
            if maxT1Idx < len(indices):  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx + mbrg :])
                )

            cur_train_mask = np.zeros(len(indices), dtype=bool)
            cur_train_mask[train_indices] = True

            train_mask = train_mask & cur_train_mask

        test_index = reduce(lambda x, y: np.concatenate((x, y)), test_indices.values())

        return indices[train_mask], test_index

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X: (pd.DataFrame) Samples dataset that is to be split.
        :param y: (pd.Series) Sample labels series.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        if (X.index == self.samples_info_sets.index).sum() != len(
            self.samples_info_sets
        ):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)

        splits = [test_ids for _, test_ids in super().split(X, y, groups)]
        splits_indices = dict(zip(range(len(splits)), splits))

        for train_indices, test_indices in self._generate_combinatorial_test_ranges(
            splits_indices
        ):
            yield self._fill_backtest_paths(
                indices, {i: splits_indices[i] for i in test_indices}, mbrg
            )