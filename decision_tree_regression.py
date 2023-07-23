import numpy as np
from sklearn.datasets import load_diabetes


class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(max_leafs, 2)
        self.leafs_cnt = 1
        self.tree = None
        self.bins = bins
        self.separators = {}

    def create_separators(self, arr):
        unique_values = np.sort(np.unique(arr))
        return [
            (left + right) / 2
            for left, right in zip(unique_values[:-1], unique_values[1:])
        ]

    def calc_mse(self, y):
        return np.mean(np.square(y - np.mean(y)))

    def find_gain(self, X, y, col, separator):
        split_idxs = X[X[col] <= separator].index
        left_idxs = split_idxs
        right_idxs = [idx for idx in X.index if idx not in left_idxs]
        y_left = y.loc[left_idxs]
        y_right = y.loc[right_idxs]
        mse_parent = self.calc_mse(y)
        mse_left = self.calc_mse(y_left)
        mse_right = self.calc_mse(y_right)
        return mse_parent - (
            y_left.shape[0] / y.shape[0] * mse_left
            + y_right.shape[0] / y.shape[0] * mse_right
        )

    def get_best_split(self, X, y):
        best_gain = -float("inf")
        best_col = None
        best_split_value = None
        for col in X.columns:
            if not self.bins:
                separators = self.create_separators(X[col].values)
            else:
                separators = self.separators[col]
            for separator in separators:
                gain = self.find_gain(X, y, col, separator)
                if gain > best_gain:
                    best_gain = gain
                    best_split_value = separator
                    best_col = col
        return best_col, best_split_value, best_gain

    def is_leaf(self, X, depth):
        if X.shape[0] == 1:
            return True
        if depth == self.max_depth:
            return True
        if X.shape[0] < self.min_samples_split:
            return True
        if self.leafs_cnt >= self.max_leafs:
            return True

    def build_tree(self, X, y, depth):
        if self.is_leaf(X, depth):
            return {
                "is_leaf": True,
                "y_values": y,
                "y_count": y.shape[0],
                "y_mean": np.mean(y),
                "count": y.shape[0],
            }
        self.leafs_cnt += 1
        best_col, best_split_value, best_gain = self.get_best_split(X, y)
        left_idxs = X[X[best_col] <= best_split_value].index
        right_idxs = [idx for idx in X.index if idx not in left_idxs]
        left_node = self.build_tree(X.loc[left_idxs], y.loc[left_idxs], depth=depth + 1)
        right_node = self.build_tree(
            X.loc[right_idxs], y.loc[right_idxs], depth=depth + 1
        )
        return {
            "is_leaf": False,
            "y_count": y.shape[0],
            "left_node": left_node,
            "right_node": right_node,
            "split_col": best_col,
            "split_gain": best_gain,
            "split_value": best_split_value,
            "y_values": y,
        }

    def calculate_feature_importances(self):
        def calculate_feature_importances_for_node(node):
            if node["is_leaf"]:
                return 0
            left_weight = node["left_node"]["y_count"] / node["y_count"]
            left_mse = self.calc_mse(node["left_node"]["y_values"])
            right_mse = self.calc_mse(node["right_node"]["y_values"])
            right_weight = node["right_node"]["y_count"] / node["y_count"]
            source_mse = self.calc_mse(node["y_values"])
            source_weight = node["y_count"] / self.x_shape
            self.fi[node["split_col"]] = self.fi.get(node["split_col"], 0) + (
                source_weight
            ) * (source_mse - (left_weight * left_mse) - (right_weight * right_mse))
            calculate_feature_importances_for_node(node['left_node'])
            calculate_feature_importances_for_node(node['right_node'])
        calculate_feature_importances_for_node(self.tree)

    def fit(self, X, y):
        self.x_shape = X.shape[0]
        self.fi = {col: 0 for col in X.columns}
        if self.bins:
            for column in X.columns:
                col_separators = self.create_separators(X[column].values)
                if len(col_separators) <= self.bins - 1:
                    self.separators[column] = col_separators
                else:
                    _, self.separators[column] = np.histogram(
                        X[column].values, bins=self.bins
                    )
        self.tree = self.build_tree(X, y, depth=0)
        self.calculate_feature_importances()

    def find_value(self, row, node):
        if node["is_leaf"]:
            return node["y_mean"]
        if row[node["split_col"]] <= node["split_value"]:
            return self.find_value(row, node["left_node"])
        else:
            return self.find_value(row, node["right_node"])

    def predict(self, X):
        result = []
        for idx, row in X.iterrows():
            result.append(self.find_value(row, self.tree))
        return np.array(result)

    def print_tree(self):
        def print_node(node, indent):
            if node["is_leaf"]:
                print(indent + "leaf = " + str(node["y_mean"]))
                return
            print(indent + node["split_col"] + " > " + str(node["split_value"]))
            left_node = node["left_node"]
            right_node = node["right_node"]
            print_node(left_node, indent=indent + "  ")
            print_node(right_node, indent=indent + "  ")

        print_node(self.tree, "")

    def __repr__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"


if __name__ == "__main__":
    data = load_diabetes(as_frame=True)
    X, y = data["data"], data["target"]

    model = MyTreeReg(max_depth=3, min_samples_split=2, max_leafs=1)
    model.fit(X, y)
    # model.print_tree()
    print(model.predict(X))
