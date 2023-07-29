import random
from sklearn.datasets import load_diabetes, make_regression
import pandas as pd
import numpy as np
from decision_tree_regression import MyTreeReg


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

    def fit(self, X, y):
        random.seed(self.random_state)
        self.fi = {column: 0 for column in X.columns}
        for _ in range(self.n_estimators):
            cols_sample_count = int(round(len(X.columns) * self.max_features))
            rows_sample_count = int(round(X.shape[0] * self.max_samples))
            cols_idx = random.sample(list(X.columns), cols_sample_count)
            rows_idx = random.sample(range(X.shape[0]), rows_sample_count)
            X_batch = X.loc[rows_idx][cols_idx]
            y_batch = y.loc[rows_idx]
            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins,
                rows_count=X.shape[0]
            )
            tree.fit(X_batch, y_batch)
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt
        for tree in self.trees:
            for col, importance in tree.fi.items():
                self.fi[col] = self.fi.get(col, 0) + importance

    def predict(self, X):
        result = []
        for i in range(X.shape[0]):
            row = X.iloc[i:i+1]
            tree_predicts = [tree.predict(row) for tree in self.trees]
            result.append(np.mean(tree_predicts))
        return np.array(result)

    def __repr__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}"


if __name__ == "__main__":
    X, y = make_regression(n_samples=150, n_features=14, n_informative=10, noise=15,
                           random_state=42)
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    params = {"n_estimators": 5, "max_depth": 4, "max_features": 0.4, "max_samples": 0.3}
    model = MyForestReg(**params)
    model.fit(X, y)
    print(model.fi)
