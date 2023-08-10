class MyLogReg:
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
        self.l1 = l1_coef
        self.l2 = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def calc_metric(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if self.metric == "accuracy":
            return np.sum(y_true == y_pred) / y_true.shape[0]
        elif self.metric == "precision":
            return precision
        elif self.metric == "recall":
            return recall
        elif self.metric == "f1":
            return 2 * precision * recall / (precision + recall)
        elif self.metric == "roc_auc":
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

    def calc_loss(self, y_true, y_pred):
        eps = 1e-15
        if self.reg == "l1":
            return -np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            ) + self.l1 * np.sum(np.abs(self.weights))
        elif self.reg == "l2":
            return -np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            ) + self.l2 * np.sum(np.square(self.weights))
        elif self.reg == "elasticnet":
            return (
                -np.mean(
                    y_true * np.log(y_pred + eps)
                    + (1 - y_true) * np.log(1 - y_pred + eps)
                )
                + self.l1 * np.sum(np.abs(self.weights))
                + self.l2 * np.sum(np.square(self.weights))
            )
        else:
            return -np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            )

    def calc_grad(self, X, y_true, y_pred):
        if self.reg == "l1":
            grad = np.dot(np.transpose(y_pred - y_true), X) / X.shape[
                0
            ] + self.l1 * np.sign(self.weights)
        elif self.reg == "l2":
            grad = (
                np.dot(np.transpose(y_pred - y_true), X) / X.shape[0]
                + self.l2 * 2 * self.weights
            )
        elif self.reg == "elasticnet":
            grad = (
                np.dot(np.transpose(y_pred - y_true), X) / X.shape[0]
                + self.l1 * np.sign(self.weights)
                + self.l2 * 2 * self.weights
            )
        else:
            grad = np.dot(np.transpose(y_pred - y_true), X) / X.shape[0]
        return grad

    def calc_learning_rate(self, iteration):
        if isinstance(self.learning_rate, int) or isinstance(self.learning_rate, float):
            return self.learning_rate
        return self.learning_rate(iteration)

    def get_batch(self, X, y):
        if isinstance(self.sgd_sample, int):
            sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            return X.iloc[sample_rows_idx], y.iloc[sample_rows_idx]
        elif isinstance(self.sgd_sample, float):
            sample_size = int(X.shape[0] * self.sgd_sample)
            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            return X.iloc[sample_rows_idx], y.iloc[sample_rows_idx]
        else:
            return X, y

    def fit(self, X_train, y_train, verbose=False):
        random.seed(self.random_state)
        X = X_train.copy()
        X["intercept"] = np.ones(X.shape[0])
        self.weights = np.ones(X.shape[1])
        for iter in range(self.n_iter):
            X_batch, y_batch = self.get_batch(X, y_train)
            predictions = 1 / (1 + np.exp(-np.dot(X_batch, self.weights)))
            loss = self.calc_loss(y_train, 1 / (1 + np.exp(-np.dot(X, self.weights))))
            grad = self.calc_grad(X_batch, y_batch, predictions)
            self.weights -= self.calc_learning_rate(iter + 1) * grad
            if verbose and (iter % verbose == 0) and not self.metric:
                print(f"{iter} | loss: {loss}")
                continue

            if self.metric == "roc_auc":
                metric = self.calc_metric(y_train, self.predict_proba(X))
            else:
                metric = self.calc_metric(y_train, self.predict(X))
            self.best_score = metric
            if verbose and (iter % verbose == 0):
                print(f"{iter} | loss: {loss} | {self.metric}: {metric}")

    def predict_proba(self, X_test):
        X = X_test.copy()
        X["intercept"] = np.ones(X.shape[0])
        predictions = 1 / (1 + np.exp(-np.dot(X, self.weights)))
        return predictions

    def predict(self, X_test):
        predictions = self.predict_proba(X_test)
        return np.where(predictions > 0.5, 1, 0)

    def get_coef(self):
        return self.weights[:-1]

    def get_best_score(self):
        return self.best_score

    def __repr__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )
