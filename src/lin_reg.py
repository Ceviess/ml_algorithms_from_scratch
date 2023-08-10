"""
lin_reg.py - Custom Linear Regression Model

This module contains the `MyLineReg` class which implements a custom linear regression model.

The class provides methods for training and predicting using linear regression, as well as
utility functions to calculate various metrics and regularization techniques.

Attributes:
    MyLineReg: A class that implements a custom linear regression model.

Usage:
    from lin_reg import MyLineReg

    # Instantiate the linear regression model
    model = MyLineReg(n_iter=100, learning_rate=0.1, metric="rmse", reg="l2")

    # Fit the model on training data
    model.fit(training_data, target_variable, verbose=True)

    # Make predictions using the trained model
    predictions = model.predict(test_data)

    # Get the best achieved metric score during training
    best_score = model.get_best_score()

    # Get the learned coefficients of the model
    coefficients = model.get_coef()
"""

import numpy as np
import pandas as pd
from numpy import random


class MyLineReg:
    """
    Custom Linear Regression Model

    This class implements a custom linear regression model with support for various
    metrics, regularization techniques, and stochastic gradient descent.

    Attributes:
        n_iter (int): Number of training iterations.
        learning_rate (float or callable): Learning rate for gradient descent. Can be a
            float or a callable function that takes an iteration number and returns a float.
        weights (numpy.ndarray): Coefficients of the linear regression model.
        metric (str): Evaluation metric for model performance during training. Supported
            metrics are "mae" (Mean Absolute Error), "mse" (Mean Squared Error),
            "rmse" (Root Mean Squared Error), "mape" (Mean Absolute Percentage Error),
            and "r2" (Coefficient of Determination).
        reg (str or None): Regularization type. Supported options are "l1" (L1 regularization),
            "l2" (L2 regularization), and None (no regularization).
        l1_coef (float): Coefficient for L1 regularization.
        l2_coef (float): Coefficient for L2 regularization.
        sgd_sample (int or float or None): Size of mini-batch or subsample for stochastic
            gradient descent. Can be an integer (number of samples) or a float (fraction
            of total samples), or None for full batch gradient descent.
        random_state (int): Seed for random number generation.

    Methods:
        calc_mae: Calculate Mean Absolute Error between true and predicted values.
        calc_mse: Calculate Mean Squared Error between true and predicted values.
        calc_rmse: Calculate Root Mean Squared Error between true and predicted values.
        calc_r2: Calculate Coefficient of Determination (R-squared) between true and
            predicted values.
        calc_mape: Calculate Mean Absolute Percentage Error between true and predicted
            values.
        calc_loss: Calculate the loss function based on the selected regularization.
        calc_grad: Calculate gradients of the loss function for gradient descent.
        calc_learning_rate: Calculate the learning rate for a given iteration.
        get_batch: Get mini-batch or subsample of data for stochastic gradient descent.
        fit: Train the linear regression model on input data.
        predict: Generate predictions using the trained linear regression model.
        get_best_score: Get the best achieved metric score during training.
        get_coef: Get the learned coefficients of the linear regression model.

    Usage:
        model = MyLineReg(n_iter=100, learning_rate=0.1, metric="rmse", reg="l2")
        model.fit(training_data, target_variable, verbose=True)
        predictions = model.predict(test_data)
        best_score = model.get_best_score()
        coefficients = model.get_coef()

    """

    def __init__(
        self,
        n_iter=100,
        learning_rate=0.1,
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.calc_metric = {
            "mae": self.calc_mae,
            "mse": self.calc_mse,
            "rmse": self.calc_rmse,
            "mape": self.calc_mape,
            "r2": self.calc_r2,
            None: lambda x, y: None,
        }
        self.best_score = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.random_state = random_state
        self.sample_size = sgd_sample

    def calc_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
        Calculate the Mean Absolute Error (MAE) between true values and predicted values.

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated Mean Absolute Error.
        :rtype: np.float64
        """
        return np.mean(np.abs(y_pred - y_true))

    def calc_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
        Calculate the Mean Squared Error (MSE) between true values and predicted values.

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated Mean Squared Error.
        :rtype: np.float64
        """
        return np.mean(np.square(y_true - y_pred))

    def calc_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
        Calculate the Root Mean Squared Error (RMSE) between true values and predicted values.

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated Root Mean Squared Error.
        :rtype: np.float64
        """
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    def calc_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the coefficient of determination (R^2) between true values and predicted values.

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated R^2 coefficient.
        :rtype: float
        """
        return 1 - (
            np.sum(np.square(y_true - y_pred))
            / np.sum(np.square(y_true - np.mean(y_true)))
        )

    def calc_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE) between true values and predicted values

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated MAPE.
        :rtype: float
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def calc_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss function based on predicted values and true values,
            considering regularization.

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param y_pred: The predicted values.
        :type y_pred: np.ndarray

        :return: The calculated loss.
        :rtype: float
        """
        loss = self.calc_mse(y_true, y_pred)
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
        self, data: np.ndarray, y_true: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to model parameters.

        :param data: The feature matrix.
        :type data: np.ndarray

        :param y_true: The true values.
        :type y_true: np.ndarray

        :param predictions: The predicted values.
        :type predictions: np.ndarray

        :return: The calculated gradient.
        :rtype: np.ndarray
        """
        n_rows = y_true.shape[0]
        delta = np.transpose(predictions - y_true)
        grad = 2 / n_rows * np.dot(delta, data)
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

    def calc_learning_rate(self, iteration: any) -> float:
        """
        Calculate the learning rate for a specific iteration.

        :param iteration: The current iteration.
        :type iteration: any

        :return: The calculated learning rate.
        :rtype: float
        """
        if isinstance(self.learning_rate, (float, int)):
            return self.learning_rate
        return self.learning_rate(iteration)

    def get_batch(self, data: pd.DataFrame, labels: pd.Series) -> tuple:
        """
        Get a batch of data samples from the given feature matrix and target values.

        :param data: The feature matrix.
        :type data: pd.DataFrame

        :param labels: The target values.
        :type labels: pd.Series

        :return: A tuple containing the batched feature matrix and target values.
        :rtype: tuple
        """
        if isinstance(self.sample_size, int):
            sample_rows_idx = random.sample(range(data.shape[0]), self.sample_size)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        if isinstance(self.sample_size, float):
            sample_size = int(data.shape[0] * self.sample_size)
            sample_rows_idx = random.sample(range(data.shape[0]), sample_size)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        return data, labels

    def fit(
        self, train_data: pd.DataFrame, labels: np.ndarray, verbose: bool = False
    ) -> None:
        """
        Fit the model using the given feature matrix and target values.

        :param train_data: The feature matrix.
        :type train_data: pd.DataFrame

        :param labels: The target values.
        :type labels: np.ndarray

        :param verbose: Whether to print training progress, defaults to False.
        :type verbose: bool, optional
        """
        random.seed(self.random_state)

        data = train_data.copy()
        n_rows = data.shape[0]
        n_features = data.shape[1]
        data["intercept"] = [1] * n_rows
        self.weights = np.ones(n_features + 1)

        for iteration in range(self.n_iter):
            data_batch, y_batch = self.get_batch(data, labels)
            predictions = np.dot(data_batch, self.weights)
            loss = self.calc_loss(labels, np.dot(data, self.weights))
            grad = self.calc_grad(data_batch, y_batch, predictions)
            self.weights -= self.calc_learning_rate(iteration + 1) * grad
            updated_predictions = np.dot(data, self.weights)
            metric_value = self.calc_metric[self.metric](labels, updated_predictions)
            self.best_score = metric_value
            if not verbose:
                continue
            if iteration % verbose == 0:
                print(f"{iteration} | loss: {loss} | {self.metric}: {metric_value}")

    def predict(self, predict_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        :param predict_data: The feature matrix for prediction.
        :type predict_data: pd.DataFrame

        :return: Predicted target values.
        :rtype: np.ndarray
        """
        data = predict_data.copy()
        data["intercept"] = [1] * predict_data.shape[0]
        predictions = np.dot(data, self.weights)
        return predictions

    def get_best_score(self) -> float:
        """
        Get the best score achieved during training.

        :return: The best score achieved.
        :rtype: float
        """
        return self.best_score

    def get_coef(self) -> np.ndarray:
        """
        Get the coefficients of the trained model.

        :return: Coefficients of the model.
        :rtype: np.ndarray
        """
        return self.weights[:-1]

    def __repr__(self):
        return (
            f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )
