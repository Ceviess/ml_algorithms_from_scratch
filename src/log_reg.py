"""
log_reg.py - Custom Logistic Regression Implementation

This module provides a custom implementation of Logistic Regression for educational purposes.
The `MyLogReg` class allows users to train a logistic regression model with various hyperparameters,
including learning rate, regularization, and more.

Classes:
- MyLogReg: Custom Logistic Regression implementation.

Usage:
1. Import the module: `from log_reg import MyLogReg`
2. Create an instance of the `MyLogReg` class: `model = MyLogReg()`
3. Fit the model using training data: `model.fit(X_train, y_train)`
4. Make predictions using the trained model: `predictions = model.predict(X_test)`
5. Access model attributes: `best_score = model.get_best_score()`
"""

import random
import numpy as np


class MyLogReg:
    """
    Custom Logistic Regression implementation.

    This class provides a custom implementation of Logistic Regression.
    It allows you to set various hyperparameters such as the number of iterations,
    learning rate, regularization, and more.

    Parameters:
    - n_iter (int): Number of iterations for training, default is 100.
    - learning_rate (float): Learning rate for gradient descent, default is 0.1.
    - weights (array-like): Initial weights for the model, default is None.
    - metric (str): Metric to evaluate the model's performance, default is None.
    - reg (str): Regularization type ("l1", "l2", or None), default is None.
    - l1_coef (float): Coefficient for L1 regularization, default is None.
    - l2_coef (float): Coefficient for L2 regularization, default is None.
    - sgd_sample (int or float): Sample size for Stochastic Gradient Descent, default is None.
    - random_state (int): Seed for random number generation, default is 42.

    Attributes:
    - n_iter (int): Number of iterations for training.
    - learning_rate (float): Learning rate for gradient descent.
    - weights (array-like): Model weights.
    - metric (str): Metric used for model evaluation.
    - best_score (float): Best score achieved during training.
    - reg (str): Regularization type ("l1", "l2", or None).
    - l1 (float): Coefficient for L1 regularization.
    - l2 (float): Coefficient for L2 regularization.
    - sgd_sample (int or float): Sample size for Stochastic Gradient Descent.
    - random_state (int): Seed for random number generation.

    Methods:
    - fit(df, y, verbose=False): Fit the model using the given data.
    - predict(df): Make predictions using the trained model.
    - get_best_score(): Get the best score achieved during training.
    - get_coef(): Get the coefficients of the trained model.
    """

    def __init__(
        self,
        n_iter=100,
        learning_rate=0.1,
        weights=None,
        metric=None,
        reg=None,
        l1_coef=None,
        l2_coef=None,
        sgd_sample=None,
        random_state=42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def calc_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the specified metric based on true values and predicted values.

        Parameters:
        :param y_true: The true values.
        :type: np.ndarray
        :param y_pred: The predicted values.
        :type: np.ndarray
        :return: The calculated metric value.
        :rtype: float
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        false_negatives = np.sum((y_pred == 0) & (y_true == 1))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        if self.metric == "accuracy":
            return np.sum(y_true == y_pred) / y_true.shape[0]
        if self.metric == "precision":
            return precision
        if self.metric == "recall":
            return recall
        if self.metric == "f1":
            return 2 * precision * recall / (precision + recall)
        if self.metric == "roc_auc":
            y_pred = np.round(y_pred, 10)
            score_sorted = sorted(zip(y_true, y_pred), key=lambda x: x[1])
            ranked = 0
            for i in range(len(score_sorted) - 1):
                cur_true = score_sorted[i][0]
                if cur_true == 1:
                    continue
                for j in range(i + 1, len(score_sorted)):
                    if (
                        score_sorted[j][0] == 1
                        and score_sorted[j][1] == score_sorted[i][1]
                    ):
                        ranked += 0.5
                    elif score_sorted[j][0] == 1:
                        ranked += 1
            return (
                ranked
                / np.sum(np.where(y_true == 1, 1, 0))
                / np.sum(np.where(y_true == 0, 1, 0))
            )

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss function for the given true values and predicted values.

        :param y_true: The true values.
        :type y_true: np.ndarray
        :param y_pred : The predicted values.
        :type y_pred: np.ndarray
        :return: The calculated loss value.
        :rtype: float
        """
        eps = 1e-15
        loss = -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )
        if self.reg == "l1":
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        if self.reg == "l2":
            loss += self.l2_coef * np.sum(np.square(self.weights))
        if self.reg == "elasticnet":
            loss = (
                loss
                + self.l1_coef * np.sum(np.abs(self.weights))
                + self.l2_coef * np.sum(np.square(self.weights))
            )
        return loss

    def calc_grad(
        self, data: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to the weights.

        :param data: The feature matrix.
        :type data: np.ndarray

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated gradient vector.
        :rtype: np.ndarray
        """
        grad = np.dot(np.transpose(y_pred - y_true), data) / data.shape[0]
        if self.reg == "l1":
            grad += self.l1_coef * np.sign(self.weights)
        if self.reg == "l2":
            grad += self.l2_coef * 2 * self.weights
        if self.reg == "elasticnet":
            grad = (
                grad
                + self.l1_coef * np.sign(self.weights)
                + self.l2_coef * 2 * self.weights
            )
        return grad

    def calc_learning_rate(self, iteration: int) -> float:
        """
        Calculate the learning rate for the current iteration.

        :param iteration: The current iteration.
        :type iteration: int

        :return: The calculated learning rate.
        :rtype: float
        """
        if isinstance(self.learning_rate, (int, float)):
            return self.learning_rate
        return self.learning_rate(iteration)

    def get_batch(self, data, labels):
        """
        Get a batch of data samples for Stochastic Gradient Descent (SGD).

        :param data: The feature matrix.
        :type data: np.ndarray

        :param labels: The target values.
        :type labels: np.ndarray

        :return: A batch of feature matrix and target values.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if isinstance(self.sgd_sample, int):
            sample_rows_idx = random.sample(range(data.shape[0]), self.sgd_sample)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        if isinstance(self.sgd_sample, float):
            sample_size = int(data.shape[0] * self.sgd_sample)
            sample_rows_idx = random.sample(range(data.shape[0]), sample_size)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        return data, labels

    def fit(self, data_train: np.ndarray, y_train: np.ndarray, verbose: bool = False):
        """
        Fit the model using the given training data.

        :param data_train: The feature matrix for training.
        :type data_train: np.ndarray

        :param y_train: The target values for training.
        :type y_train: np.ndarray

        :param verbose: Whether to print progress during training, defaults to False.
        :type verbose: bool
        """
        random.seed(self.random_state)
        data = data_train.copy()
        data["intercept"] = np.ones(data.shape[0])
        self.weights = np.ones(data.shape[1])
        for iter_num in range(self.n_iter):
            x_batch, y_batch = self.get_batch(data, y_train)
            predictions = 1 / (1 + np.exp(-np.dot(x_batch, self.weights)))
            loss = self.calc_loss(
                y_train, 1 / (1 + np.exp(-np.dot(data, self.weights)))
            )
            grad = self.calc_grad(x_batch, y_batch, predictions)
            self.weights -= self.calc_learning_rate(iter_num + 1) * grad
            if verbose and (iter_num % verbose == 0) and not self.metric:
                print(f"{iter_num} | loss: {loss}")
                continue

            if self.metric == "roc_auc":
                metric = self.calc_metric(y_train, self.predict_proba(data))
            else:
                metric = self.calc_metric(y_train, self.predict(data))
            self.best_score = metric
            if verbose and (iter_num % verbose == 0):
                print(f"{iter_num} | loss: {loss} | {self.metric}: {metric}")

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict the probabilities of the positive class for the given test data.

        :param x_test: The feature matrix for testing.
        :type x_test: np.ndarray

        :return: Predicted probabilities for the positive class.
        :rtype: np.ndarray
        """
        data = x_test.copy()
        data["intercept"] = np.ones(data.shape[0])
        predictions = 1 / (1 + np.exp(-np.dot(data, self.weights)))
        return predictions

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict the binary class labels for the given test data.

        :param x_test: The feature matrix for testing.
        :type x_test: np.ndarray

        :return: Predicted binary class labels (0 or 1).
        :rtype: np.ndarray
        """
        predictions = self.predict_proba(x_test)
        return np.where(predictions > 0.5, 1, 0)

    def get_coef(self):
        """
        Get the coefficients of the linear model.

        :return: Coefficients of the linear model.
        :rtype: np.ndarray
        """
        return self.weights[:-1]

    def get_best_score(self):
        """
        Get the best score achieved during training.

        :return: Best score achieved during training.
        :rtype: float
        """
        return self.best_score

    def __repr__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )
