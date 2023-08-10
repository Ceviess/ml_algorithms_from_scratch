"""
decision_tree_classification.py

This module contains the implementation of a custom decision tree classifier.

Classes:
-------
MyTreeClf
    A custom decision tree classifier that supports entropy and Gini impurity criteria for splitting
    nodes.
"""

import numpy as np
import pandas as pd


class MyTreeClf:
    """
    Custom decision tree classifier.

    Parameters:
    ----------
    max_depth : int, optional (default=5)
        The maximum depth of the tree.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    max_leafs : int, optional (default=20)
        The maximum number of leaf nodes in the tree.

    bins : int, optional (default=None)
        The number of bins to use when discretizing features.

    criterion : str, optional (default="entropy")
        The criterion used to evaluate the quality of splits ("entropy" or "gini").

    Attributes:
    ----------
    max_depth : int
        The maximum depth of the tree.

    min_samples_split : int
        The minimum number of samples required to split an internal node.

    max_leafs : int
        The maximum number of leaf nodes in the tree.

    leafs_cnt : int
        The current number of leaf nodes in the tree.

    bins : int
        The number of bins used for feature discretization.

    criterion : str
        The criterion used to evaluate the quality of splits.

    calc_criterion : dict
        A dictionary mapping criterion names to corresponding calculation functions.

    Methods:
    ----------
    calc_gini(arr)
        Calculate the Gini impurity of an array.

    calc_class(class_count, n)
        Calculate the class part of the entropy formula.

    calc_entropy(arr)
        Calculate the entropy of an array.

    get_best_split(X, y)
        Find the best split for a given dataset.

    check_leaf(X, y, depth)
        Check if a node should be a leaf.

    fit(X, y)
        Fit the decision tree classifier to the given training data.

    update_feat_importance()
        Update feature importance scores based on the trained tree.

    calc_feat_importance(cur_node_rows, y, left_mask, right_mask)
        Calculate feature importance for a node.

    build_tree(X, y, depth=0)
        Build the decision tree recursively.

    predict(X)
        Predict class labels for input data.

    print_tree()
        Print the decision tree structure.

    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=None,
        criterion="entropy",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.leafs_cnt = 1
        self.bins = bins
        self.criterion = criterion
        self.calc_criterion = {"entropy": self.calc_entropy, "gini": self.calc_gini}
        self.n_rows = None
        self.feat_importances = {}
        self.thresholds = {}
        self.tree = None

    def __repr__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def calc_gini(self, arr: np.ndarray) -> float:
        """
        Calculate the Gini impurity for a binary classification scenario.

        Gini impurity is a measure of the inequality in a distribution.
        It ranges from 0 (pure node, all instances are of the same class) to 0.5 (impure node,
        instances are evenly distributed).

        :param arr: An array of binary class labels (0 or 1).
        :type arr: np.ndarray

        :return: Gini impurity of the given class labels.
        :rtype: float
        """
        if len(arr) == 0 or len(set(arr)) == 1:
            return 0
        first_class = sum(arr) / len(arr)
        second_class = 1 - first_class
        return 1 - (np.square(first_class) + np.square(second_class))

    def calc_class(self, class_count: int, num_instances: int) -> float:
        """
        Calculate the contribution of a class to the entropy formula.

        This function calculates the value of (p_i * log2(p_i)) where p_i is the probability
        of a class occurring. It is used in entropy calculation.

        :param class_count: The count of instances belonging to a specific class.
        :type class_count: int

        :param num_instances: The total number of instances.
        :type num_instances: int

        :return: The contribution of the class to the entropy formula.
        :rtype: float
        """
        return (
            (class_count / num_instances) * np.log2(class_count / num_instances)
            if class_count != 0
            else 0
        )

    def calc_entropy(self, arr: np.ndarray) -> float:
        """
        Calculate the entropy of a binary classification scenario.

        Entropy is a measure of impurity or disorder in a set of instances.
        It ranges from 0 (pure node, all instances are of the same class) to 1 (maximum impurity).

        :param arr: An array of binary class labels (0 or 1).
        :type arr: np.ndarray

        :return: Entropy of the given class labels.
        :rtype: float
        """
        num_instances = arr.shape[0]
        pos = np.sum(arr)
        neg = num_instances - np.sum(arr)

        entropy = -(
            self.calc_class(pos, num_instances) + self.calc_class(neg, num_instances)
        )
        return entropy

    def get_best_split(
        self, data: pd.DataFrame, labels: np.ndarray
    ) -> tuple[str, float, float]:
        """
        Find the best feature and threshold for splitting the data in a decision tree node.

        This function searches for the feature and threshold that result in the highest information
        gain.

        :param data: Input features.
        :type data: pd.DataFrame

        :param labels: Class labels.
        :type labels: np.ndarray

        :return: Tuple containing the best feature, threshold, and information gain.
        :rtype: tuple[str, float, float]
        """

        def calc_gain(
            left: np.ndarray, right: np.ndarray, overall_criterion: float
        ) -> float:
            """
            Calculate the information gain based on split criteria for a decision tree node.

            Information gain measures the reduction in entropy or impurity achieved by the split.

            :param left: An array of binary class labels for the left split.
            :type left: np.ndarray

            :param right: An array of binary class labels for the right split.
            :type right: np.ndarray

            :param overall_criterion: Criterion value for the whole node before the split.
            :type overall_criterion: float

            :return: Information gain achieved by the split.
            :rtype: float
            """
            left_criterion = self.calc_criterion[self.criterion](left)
            right_criterion = self.calc_criterion[self.criterion](right)

            num_instances = left.shape[0] + right.shape[0]
            left_weight = left.shape[0] / num_instances
            right_weight = right.shape[0] / num_instances

            gain = overall_criterion - (
                left_weight * left_criterion + right_weight * right_criterion
            )
            return gain

        def get_thresholds(feat: pd.DataFrame) -> np.ndarray:
            """
            Get thresholds for feature splitting in a decision tree node.

            If self.bins is None, thresholds are calculated as the midpoints between unique feature
            values. Otherwise, thresholds are retrieved from the pre-computed self.thresholds
            dictionary.

            :param feat: A DataFrame with a single feature column.
            :type feat: pd.DataFrame

            :return: Array of thresholds for feature splitting.
            :rtype: np.ndarray
            """
            unique_values = feat[feat.columns[0]].unique()
            if not self.bins:
                thresholds = [
                    (unique_values[i] + unique_values[i + 1]) / 2
                    for i in range(unique_values.shape[0] - 1)
                ]
            else:
                thresholds = self.thresholds[feat.columns[0]]
            return thresholds

        def calc_feat(feat: pd.DataFrame) -> tuple[float, float]:
            """
            Calculate the best information gain and corresponding threshold for feature splitting.

            This function iterates through thresholds and calculates information gain for each split
            It returns the best information gain and the corresponding threshold.

            :param feat: A DataFrame with a single feature column and class labels 'y'.
            :type feat: pd.DataFrame

            :return: Tuple containing the best information gain and corresponding threshold.
            :rtype: tuple[float, float]
            """
            best_gain = -float("inf")
            best_threshold = None
            overall_criterion = self.calc_criterion[self.criterion](feat["y"].values)

            thresholds = get_thresholds(feat)
            for threshold in thresholds:
                left = feat.loc[feat[feat.columns[0]] <= threshold, "y"].values
                right = feat.loc[feat[feat.columns[0]] > threshold, "y"].values
                gain = calc_gain(left, right, overall_criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
            return best_gain, best_threshold

        dataframe = data.copy()
        dataframe["y"] = labels
        best_col = None
        best_split_value = None
        best_gain = -float("inf")

        for col in data.columns:
            feat = dataframe[[col, "y"]].sort_values(col)
            cur_gain, cur_threshold = calc_feat(feat)
            if cur_gain > best_gain:
                best_col = col
                best_split_value = cur_threshold
                best_gain = cur_gain
        return best_col, best_split_value, best_gain

    def check_leaf(self, data: pd.DataFrame, labels: np.ndarray, depth: int) -> bool:
        """
        Check if a node should be a leaf in the decision tree.

        This function determines whether a node in the decision tree should be a leaf node based
        on criteria such as the number of unique classes, maximum depth, minimum samples for
        splitting, and maximum leaf count.

        :param data: Input features.
        :type data: pd.DataFrame

        :param labels: Class labels.
        :type labels: np.ndarray

        :param depth: Depth of the current node in the tree.
        :type depth: int

        :return: True if the node should be a leaf, False otherwise.
        :rtype: bool
        """
        if len(set(labels)) == 1:
            return True
        if depth >= self.max_depth:
            return True
        if data.shape[0] < self.min_samples_split:
            return True
        if self.leafs_cnt >= self.max_leafs:
            return True
        return False

    def fit(self, x_train: pd.DataFrame, labels: np.ndarray):
        """
        Fit the decision tree classifier on the given dataset.

        This function trains the decision tree classifier by creating a tree structure using
        recursive splitting based on the features and class labels in the training dataset. It
        initializes necessary attributes and constructs the decision tree.

        :param x_train: Input features.
        :type x_train: pd.DataFrame

        :param labels: Class labels.
        :type labels: np.ndarray
        """
        self.n_rows = x_train.shape[0]
        if self.bins:
            for col in x_train.columns:
                unique_values = x_train[col].sort_values().unique()
                thresholds = [
                    (unique_values[i] + unique_values[i + 1]) / 2
                    for i in range(unique_values.shape[0] - 1)
                ]
                if len(thresholds) > (self.bins - 1):
                    _, thresholds = np.histogram(unique_values, bins=self.bins)
                    thresholds = thresholds[1:-1]
                self.thresholds[col] = thresholds
        for col in x_train.columns:
            self.feat_importances[col] = 0
        self.tree = self.build_tree(x_train, labels)
        self.update_feat_importance()

    def update_feat_importance(self):
        """
        Update feature importances after training the decision tree classifier.

        This function traverses the decision tree and accumulates the feature importances from the
        tree nodes. It recursively walks through the tree's nodes, incrementing the feature
        importances based on the node's importance value. This helps in understanding the
        significance of each feature in making classification decisions.

        Note: The function doesn't have explicit return values; it updates the feature importances
        internally.

        :return: None
        """

        def walk_tree(node):
            """
            Recursively traverse the decision tree and accumulate feature importances.

            :param node: Current node in the decision tree.
            :type node: dict
            """
            if node["leaf"]:
                return
            self.feat_importances[node["feature"]] += node["feat_importance"]
            walk_tree(node["left_child"])
            walk_tree(node["right_child"])

        walk_tree(self.tree)

    def calc_feat_importance(
        self,
        cur_node_rows: int,
        labels: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        """
        Calculate feature importance for a specific split in the decision tree.

        Feature importance is calculated as the reduction in criterion (e.g., entropy or Gini
        impurity) achieved by the split, scaled by the number of rows in the current node.

        :param cur_node_rows: Number of rows in the current node.
        :type cur_node_rows: int

        :param labels: Binary class labels for the entire node.
        :type labels: np.ndarray

        :param left_mask: Boolean mask indicating the left split.
        :type left_mask: np.ndarray

        :param right_mask: Boolean mask indicating the right split.
        :type right_mask: np.ndarray

        :return: Feature importance for the specific split.
        :rtype: float
        """
        cur_criterion = self.calc_criterion[self.criterion](labels)
        left_criterion = self.calc_criterion[self.criterion](labels[left_mask])
        right_criterion = self.calc_criterion[self.criterion](labels[right_mask])

        left_weight = labels[left_mask].shape[0] / labels.shape[0]
        right_weight = labels[right_mask].shape[0] / labels.shape[0]
        feat_importance = (
            cur_node_rows
            / self.n_rows
            * (
                cur_criterion
                - left_weight * left_criterion
                - right_weight * right_criterion
            )
        )
        return feat_importance

    def build_tree(
        self, data: pd.DataFrame, labels: np.ndarray, depth: int = 0
    ) -> dict:
        """
        Recursively build a decision tree.

        This function constructs a decision tree by recursively splitting nodes based on the best
        split found at each step. It checks for leaf conditions and returns a dictionary
        representing the constructed node.

        :param data: Feature matrix for the current node.
        :type data: pd.DataFrame

        :param labels: Binary class labels for the current node.
        :type labels: np.ndarray

        :param depth: Current depth of the tree, used for depth constraint.
        :type depth: int, optional

        :return: Dictionary representing the constructed node.
        :rtype: dict
        """
        if self.check_leaf(data, labels, depth):
            return {
                "leaf": True,
                "class": np.sum(labels) / labels.shape[0],
                "length": len(labels),
            }
        self.leafs_cnt += 1
        best_col, best_split_value, best_ig = self.get_best_split(data, labels)
        left_mask = data[best_col] <= best_split_value
        right_mask = data[best_col] > best_split_value
        left_child = self.build_tree(
            data[left_mask], labels[left_mask], depth=depth + 1
        )
        right_child = self.build_tree(
            data[right_mask], labels[right_mask], depth=depth + 1
        )
        cur_node_rows = data.shape[0]
        feat_importance = self.calc_feat_importance(
            cur_node_rows, labels, left_mask, right_mask
        )
        node = {
            "leaf": False,
            "feature": best_col,
            "split_value": best_split_value,
            "ig": best_ig,
            "left_child": left_child,
            "right_child": right_child,
            "feat_importance": feat_importance,
        }
        return node

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for a set of samples using the trained decision tree.

        This function predicts the class labels for the input samples based on the trained decision
        tree. It traverses the tree by following the decision criteria at each node until a leaf
        node is reached.

        :param data: Feature matrix for the samples to be predicted.
        :type data: pd.DataFrame

        :return: Predicted binary class labels for the input samples.
        :rtype: np.ndarray
        """

        def find_node(data: pd.Series, node: dict) -> int:
            """
            Recursively find the class label for a sample by traversing the decision tree.

            :param data: Feature vector of the sample.
            :type data: pd.Series

            :param node: Current node in the decision tree.
            :type node: dict

            :return: Predicted class label for the sample.
            :rtype: int
            """
            if node["leaf"]:
                return node["class"]
            if data[node["feature"]] <= node["split_value"]:
                return find_node(data, node["left_child"])
            return find_node(data, node["right_child"])

        result = []
        for _, row in data.iterrows():
            result.append(find_node(row, self.tree))

        return np.array(result)

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
            if node["leaf"]:
                print(indent + "leaf = " + str(node["class"]))
            else:
                print(indent + f'{node["feature"]} <= {node["split_value"]}')
                print_node(node["left_child"], indent + "\t")
                print_node(node["right_child"], indent + "\t")

        if self.tree:
            print_node(self.tree)
