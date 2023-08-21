"""
decision_tree_regression.py

This module contains the implementation of a custom decision tree regressor.

Classes:
-------
MyTreeReg
    A custom decision tree regressor
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes


def calc_mse(array: pd.Series) -> np.float64:
    """
    Calculate Mean Squared Error (MSE) of an array.

    :param array: Input array for which MSE is calculated.
    :type array: pd.Series

    :return: Calculated Mean Squared Error (MSE) value.
    :rtype: np.float64
    """
    return np.mean(np.square(array - np.mean(array)))


def create_separators(arr: np.ndarray) -> list[float]:
    """
    Create separators between unique values in an array.

    This function takes a numpy array, sorts it, and then calculates the midpoints
    between consecutive unique values. These midpoints act as separators that
    divide the range covered by the unique values into segments.

    :param arr: Input array for which separators are to be created.
    :type arr: np.ndarray

    :return: List of separator values between unique values.
    :rtype: List[float]
    """
    unique_values = np.sort(np.unique(arr))
    return [
        (left + right) / 2 for left, right in zip(unique_values[:-1], unique_values[1:])
    ]


def find_gain(
    data_sample: pd.DataFrame, target: pd.Series, col: str, separator: float
) -> float:
    """
    Calculate the reduction in Mean Squared Error (MSE) by splitting data based on a separator.

    This function calculates the reduction in MSE that would be achieved by splitting the
    data sample into two subsets based on the given separator for a specific column.

    :param data_sample: Data sample containing the feature column for splitting.
    :type data_sample: pd.DataFrame
    :param target: Target values corresponding to the data sample.
    :type target: pd.Series
    :param col: Column name to split based on.
    :type col: str
    :param separator: Value used to separate the data in the specified column.
    :type separator: float

    :return: Reduction in MSE achieved by the split.
    :rtype: float
    """
    split_idxs = data_sample[data_sample[col] <= separator].index
    left_idxs = split_idxs
    right_idxs = [idx for idx in data_sample.index if idx not in left_idxs]
    y_left = target.loc[left_idxs]
    y_right = target.loc[right_idxs]
    mse_parent = calc_mse(target)
    mse_left = calc_mse(y_left)
    mse_right = calc_mse(y_right)
    return mse_parent - (
        y_left.shape[0] / target.shape[0] * mse_left
        + y_right.shape[0] / target.shape[0] * mse_right
    )


class MyTreeReg:
    """
    Custom regression tree implementation.

    This class implements a custom regression tree for decision-making based on splitting
    data samples. It supports various hyperparameters for controlling the tree structure
    and provides methods for training, predicting, and analyzing the tree.

    :param max_depth: Maximum depth of the tree, defaults to 5.
    :param min_samples_split: Minimum number of samples required to perform a split, defaults to 2.
    :param max_leafs: Maximum number of leaf nodes, defaults to 20. Will be capped at a minimum of 2
    :param bins: Number of bins used for creating separators, defaults to None.
    :param rows_count: Number of rows in the dataset, defaults to None.

    :method get_best_split: Find the best split point for a given feature column.
    :method is_leaf: Check if a node should be a leaf node based on conditions.
    :method build_tree: Recursively build the decision tree.
    :method calculate_feature_importances: Calculate feature importances based on the tree.
    :method fit: Fit the regression tree to the training data.
    :method find_value: Find the predicted value for a given row using the tree.
    :method predict: Predict target values for a set of input data samples.
    :method print_tree: Print the structure of the decision tree.
    :method __repr__: Return a string representation of the class with hyperparameters.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Union[int, None] = None,
        rows_count: Union[int, None] = None,
    ):
        """
        Initialize the MyTreeReg instance with hyperparameters.

        :param max_depth: Maximum depth of the tree.
        :type max_depth: int, optional
        :param min_samples_split: Minimum number of samples required to perform a split.
        :type min_samples_split: int, optional
        :param max_leafs: Maximum number of leaf nodes. Will be capped at a minimum of 2.
        :type max_leafs: int, optional
        :param bins: Number of bins used for creating separators.
        :type bins: int or None, optional
        :param rows_count: Number of rows in the dataset.
        :type rows_count: int or None, optional
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(max_leafs, 2)
        self.leafs_cnt = 1
        self.tree = None
        self.bins = bins
        self.separators = {}
        self.row_counts = rows_count
        self.x_shape = None
        self.feat_importances = None

    def get_best_split(
        self, data_sample: pd.DataFrame, target: pd.Series
    ) -> tuple[str, float, float]:
        """
        Find the best split point for a given feature column.

        This method iterates over all feature columns in the data sample and, for each column,
        tries different split points to find the one that maximizes the reduction in Mean
        Squared Error (MSE) for the target values.

        :param data_sample: Data sample containing feature columns.
        :type data_sample: pd.DataFrame
        :param target: Target values corresponding to the data sample.
        :type target: pd.Series

        :return: Tuple containing the best split column name, the best split value, and the
                 corresponding gain achieved.
        :rtype: Tuple[str, float, float]
        """
        best_gain = -float("inf")
        best_col = None
        best_split_value = None
        for col in data_sample.columns:
            if not self.bins:
                separators = create_separators(data_sample[col].values)
            else:
                separators = self.separators[col]
            for separator in separators:
                gain = find_gain(data_sample, target, col, separator)
                if gain > best_gain:
                    best_gain = gain
                    best_split_value = separator
                    best_col = col
        return best_col, best_split_value, best_gain

    def is_leaf(self, data: pd.DataFrame, depth: int) -> bool:
        """
        Check if a node should be a leaf node based on conditions.

        This method examines various conditions to determine whether a node in the tree
        should be designated as a leaf node. The conditions include the number of data
        samples in the node, the current depth of the tree, the minimum samples required
        to perform a split, and the maximum number of allowed leaf nodes.

        :param data: Data sample corresponding to the node.
        :type data: pd.DataFrame
        :param depth: Current depth of the tree.
        :type depth: int

        :return: True if the node should be a leaf node, False otherwise.
        :rtype: bool
        """
        if data.shape[0] == 1:
            return True
        if depth == self.max_depth:
            return True
        if data.shape[0] < self.min_samples_split:
            return True
        if self.leafs_cnt >= self.max_leafs:
            return True

    def build_tree(self, x_values: pd.DataFrame, y_values: pd.Series, depth: int) -> dict:
        """
        Recursively build the decision tree.

        This method constructs a decision tree by recursively splitting the data sample
        into subsets based on the best available split point. It keeps track of the tree's
        structure and properties such as leaf nodes, split columns, and split gains.

        :param x_values: Data sample containing feature columns.
        :type x_values: pd.DataFrame
        :param y_values: Target values corresponding to the data sample.
        :type y_values: pd.Series
        :param depth: Current depth of the tree.
        :type depth: int

        :return: Dictionary representing the node in the tree.
        :rtype: dict
        """
        if self.is_leaf(x_values, depth):
            return {
                "is_leaf": True,
                "y_values": y_values,
                "y_count": y_values.shape[0],
                "y_mean": np.mean(y_values),
                "count": y_values.shape[0],
            }
        self.leafs_cnt += 1
        best_col, best_split_value, best_gain = self.get_best_split(x_values, y_values)
        left_idxs = x_values[x_values[best_col] <= best_split_value].index
        right_idxs = [idx for idx in x_values.index if idx not in left_idxs]
        left_node = self.build_tree(x_values.loc[left_idxs], y_values.loc[left_idxs], depth=depth + 1)
        right_node = self.build_tree(
            x_values.loc[right_idxs], y_values.loc[right_idxs], depth=depth + 1
        )
        return {
            "is_leaf": False,
            "y_count": y_values.shape[0],
            "left_node": left_node,
            "right_node": right_node,
            "split_col": best_col,
            "split_gain": best_gain,
            "split_value": best_split_value,
            "y_values": y_values,
        }

    def calculate_feature_importances(self) -> None:
        """
        Calculate feature importances based on the tree structure.

        This method calculates feature importances for the entire decision tree by
        calling a nested recursive function for each node in the tree. It updates
        the feature importance dictionary with the calculated values.

        :return: None
        """
        def calculate_feature_importances_for_node(node: dict) -> None:
            """
            Calculate feature importances for a node in the tree.

            This method recursively calculates feature importances for a node by evaluating
            the change in Mean Squared Error (MSE) due to the node's splitting decision.
            It updates the feature importance dictionary with the calculated values.

            :param node: Dictionary representing a node in the tree.
            :type node: dict

            :return: None
            """
            if node["is_leaf"]:
                return 0
            left_weight = node["left_node"]["y_count"] / node["y_count"]
            left_mse = calc_mse(node["left_node"]["y_values"])
            right_mse = calc_mse(node["right_node"]["y_values"])
            right_weight = node["right_node"]["y_count"] / node["y_count"]
            source_mse = calc_mse(node["y_values"])
            source_weight = node["y_count"] / self.x_shape
            self.feat_importances[node["split_col"]] = self.feat_importances.get(node["split_col"], 0) + (
                source_weight
            ) * (source_mse - (left_weight * left_mse) - (right_weight * right_mse))
            calculate_feature_importances_for_node(node["left_node"])
            calculate_feature_importances_for_node(node["right_node"])

        calculate_feature_importances_for_node(self.tree)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the regression tree to the training data.

        This method prepares the data, initializes instance variables, constructs the
        decision tree, and calculates feature importances.

        :param x_train: Training data sample containing feature columns.
        :type x_train: pd.DataFrame
        :param y_train: Target values corresponding to the training data.
        :type y_train: pd.Series

        :return: None
        """
        self.x_shape = self.row_counts if self.row_counts else x_train.shape[0]
        self.feat_importances = {col: 0 for col in x_train.columns}
        if self.bins:
            separators = [create_separators(x_train[column].values) for column in x_train.columns]
            self.separators = {
                column: sep
                if len(sep) <= (self.bins - 1)
                else np.histogram(x_train[column].values, bins=self.bins)[1][1:-1]
                for sep, column in zip(separators, x_train.columns)
            }

        self.tree = self.build_tree(x_train, y_train, depth=0)
        self.calculate_feature_importances()

    def find_value(self, row: pd.Series, node: dict) -> float:
        """
        Traverse the decision tree to predict a target value.

        This method recursively navigates the decision tree based on the input feature values
        of a sample and returns the predicted target value for the leaf node reached.

        :param row: Feature values of the input sample.
        :type row: pd.Series
        :param node: Dictionary representing a node in the tree.
        :type node: dict

        :return: Predicted target value for the input sample.
        :rtype: float
        """
        if node["is_leaf"]:
            return node["y_mean"]
        if row[node["split_col"]] <= node["split_value"]:
            return self.find_value(row, node["left_node"])
        return self.find_value(row, node["right_node"])

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """
        Predict target values using the trained regression tree.

        This method predicts target values for a given set of input samples by traversing
        the decision tree and returning the predicted target values for each sample.

        :param x_test: Input data samples containing feature columns.
        :type x_test: pd.DataFrame

        :return: Predicted target values for the input samples.
        :rtype: np.ndarray
        """
        return np.array([self.find_value(row, self.tree) for _, row in x_test.iterrows()])

    def print_tree(self) -> None:
        """
        Print a textual representation of the trained decision tree.

        This function prints the structure of the decision tree, displaying the nodes and splits
        along with their associated feature and split value. Leaf nodes are labeled with their
        predicted class.

        :return: None
        :rtype: None
        """
        def print_node(node: dict, indent: str = "") -> None:
            """
            Recursively print the structure of a decision tree node.

            :param node: Current node in the decision tree.
            :type node: dict

            :param indent: Indentation string for visual hierarchy.
            :type indent: str

            :return: None
            :rtype: None
            """
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
    dataset = load_diabetes(as_frame=True)
    X, y = dataset["data"], dataset["target"]

    model = MyTreeReg(max_depth=5, min_samples_split=5, max_leafs=10)
    model.fit(X, y)
    # model.print_tree()
    print(model.predict(X))
