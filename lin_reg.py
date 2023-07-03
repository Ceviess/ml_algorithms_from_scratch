class MyLineReg():
    
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
        self.metric_dict = {
            'mae': self.calc_mae,
            'mse': self.calc_mse,
            'rmse': self.calc_rmse,
            'mape': self.calc_mape,
            'r2': self.calc_r2,
            None: lambda x, y: None,
        }
        self.best_score = None
        self.l1 = l1_coef
        self.l2 = l2_coef
        self.reg = reg
        self.random_state = random_state
        self.sample_size = sgd_sample

    def calc_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    def calc_mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def calc_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    def calc_r2(self, y_true, y_pred):
        return 1 - (np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))

    def calc_mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def calc_loss(self, y, predictions):
        if not self.reg:
            loss = np.mean(np.square(predictions - y))
        elif self.reg == 'l1':
            loss = np.mean(np.square(predictions - y)) + self.l1 * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            loss = np.mean(np.square(predictions - y)) + self.l2 * np.sum(np.square(self.weights))
        else:
            loss = np.mean(np.square(predictions - y)) + self.l1 * np.sum(np.abs(self.weights)) + self.l2 * np.sum(np.square(self.weights))
        return loss
    
    def calc_grad(self, X, y, predictions):
        n_rows = y.shape[0]
        if not self.reg:
            grad = 2 / n_rows * np.dot(np.transpose(predictions - y), X) 
        elif self.reg == 'l1':
            grad = 2 / n_rows * np.dot(np.transpose(predictions - y), X) + self.l1 * np.sign(self.weights)
        elif self.reg == 'l2':
            grad = 2 / n_rows * np.dot(np.transpose(predictions - y), X) + self.l2 * 2 * self.weights
        else:
            grad = 2 / n_rows * np.dot(np.transpose(predictions - y), X) + self.l1 * np.sign(self.weights) + self.l2 * 2 * self.weights
        return grad
    
    def calc_learning_rate(self, iteration):
        if type(self.learning_rate) == int or type(self.learning_rate) == float:
            return self.learning_rate
        return self.learning_rate(iteration)
    
    def get_batch(self, X, y):
        if type(self.sample_size) == int:
            sample_rows_idx = random.sample(range(X.shape[0]), self.sample_size)
            return X.iloc[sample_rows_idx], y.iloc[sample_rows_idx]
        elif type(self.sample_size) == float:
            sample_size = int(X.shape[0] * self.sample_size)
            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            return X.iloc[sample_rows_idx], y.iloc[sample_rows_idx]
        else:
            return X, y
    
    def fit(self, df, y, verbose=False):
        random.seed(self.random_state)
        
        X = df.copy()
        n_rows = X.shape[0]
        n_features = X.shape[1]
        X['intercept'] = [1] * n_rows
        self.weights = np.ones(n_features + 1)
        
        for iter in range(self.n_iter):
            X_batch, y_batch = self.get_batch(X, y)
            predictions = np.dot(X_batch, self.weights)
            loss = self.calc_loss(y, np.dot(X, self.weights))
            grad = self.calc_grad(X_batch, y_batch, predictions)
            self.weights -= self.calc_learning_rate(iter + 1) * grad
            updated_predictions = np.dot(X, self.weights)
            metric_value = self.metric_dict[self.metric](y, updated_predictions)
            self.best_score = metric_value
            if not verbose:
                continue
            if iter % verbose == 0:
                print(f"{iter} | loss: {loss} | {self.metric}: {metric_value}")
                
    def predict(self, df):
        X = df.copy()
        X['intercept'] = [1] * df.shape[0]
        predictions = np.dot(X, self.weights)
        return predictions

    def get_best_score(self):
        return self.best_score
                
    def get_coef(self):
        return self.weights[:-1]
            
    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
