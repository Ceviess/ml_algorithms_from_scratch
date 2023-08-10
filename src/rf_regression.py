import random
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from decision_tree_regression import MyTreeReg


def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def calc_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def calc_r2(y_true, y_pred):
    return 1 - (
        np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    )


def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class MyForestReg:
    def __init__(
        self,
        n_estimators=10,
        max_features=0.5,
        max_samples=0.5,
        random_state=42,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        oob_score=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = None
        self.oob_metrics = []

    def calculate_oob_score(self, y_true, y_pred):
        metrics = {
            "mae": calc_mae,
            "mse": calc_mse,
            "rmse": calc_rmse,
            "mape": calc_mape,
            "r2": calc_r2,
            None: lambda x, y: 0,
        }
        return metrics[self.oob_score](y_true, y_pred)

    def fit(self, X, y):
        random.seed(self.random_state)
        self.fi = {column: 0 for column in X.columns}
        for _ in range(self.n_estimators):
            cols_sample_count = int(round(len(X.columns) * self.max_features))
            rows_sample_count = int(round(X.shape[0] * self.max_samples))
            cols_idx = random.sample(list(X.columns), cols_sample_count)
            rows_idx = random.sample(range(X.shape[0]), rows_sample_count)
            oob_idx = [idx for idx in X.index if idx not in rows_idx]

            X_batch = X.loc[rows_idx][cols_idx]
            X_oob = X.loc[oob_idx][cols_idx]
            y_batch = y.loc[rows_idx]
            y_oob = y.loc[oob_idx]

            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins,
                rows_count=X.shape[0],
            )
            tree.fit(X_batch, y_batch)
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            oob_predict = tree.predict(X_oob)
            self.oob_metrics.append(self.calculate_oob_score(y_oob, oob_predict))

        self.oob_score_ = sum(self.oob_metrics) / len(self.oob_metrics)
        for tree in self.trees:
            for col, importance in tree.fi.items():
                self.fi[col] = self.fi.get(col, 0) + importance

    def predict(self, X):
        tree_predicts = np.mean([tree.predict(X) for tree in self.trees], axis=0)
        return tree_predicts

    def __repr__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}"


if __name__ == "__main__":
    X, y = make_regression(
        n_samples=150, n_features=14, n_informative=10, noise=15, random_state=42
    )
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f"col_{col}" for col in X.columns]

    params = {
        "n_estimators": 5,
        "max_depth": 4,
        "max_features": 0.4,
        "max_samples": 0.3,
        "oob_score": "mae",
    }
    model = MyForestReg(**params)
    model.fit(X, y)
    print(model.predict(X.iloc[:10]))
