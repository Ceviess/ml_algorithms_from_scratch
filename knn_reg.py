class MyKNNReg():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight
        self.calc_metric = {
            "euclidean": self.euclidean,
            "chebyshev": self.chebyshev,
            "manhattan": self.manhattan,
            "cosine": self.cosine,
        }
        self.calc_distance = {
            "uniform": self.calc_uniform_weights,
            "rank": self.calc_rank_weights,
            "distance": self.calc_distance_weights,
        }
        
    def euclidean(self, row1, row2):
        return np.sqrt(np.sum(np.square(row1 - row2)))
    
    def chebyshev(self, row1, row2):
        return np.max(np.abs(row1 - row2))
    
    def manhattan(self, row1, row2):
        return np.sum(np.abs(row1 - row2))
    
    def cosine(self, row1, row2):
        return 1 - (np.sum(row1 * row2) / (np.sqrt(np.sum(np.square(row1))) * np.sqrt(np.sum(np.square(row2)))))
        
    def __repr__(self):
        return f"MyKNNReg class: k={self.k}"
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])
        
    def calc_rank_weights(self, neighbors_idxs):
        denom = np.sum(1 / np.arange(1, self.k + 1))
        result = []
        for idx, row in enumerate(neighbors_idxs):
            y = self.y.loc[row]
            result.append(np.sum([(1 / (i + 1)) * val / denom for i, val in enumerate(y)]))
        return result
    
    def calc_distance_weights(self, neighbors_idxs):
        result = []
        for idx, neighbors in enumerate(neighbors_idxs):
            y = self.y.loc[neighbors]
            distances = np.array([self.dist_matrix[idx][n][0] for n in neighbors])
            denom = np.sum(1 / distances)
            result.append(np.sum([((1 / distances[i]) * val) / denom for i, val in enumerate(y)]))
        return result
    
    def calc_uniform_weights(self, neighbors_idxs):
        result = []
        for idx, neighbors in enumerate(neighbors_idxs):
            y = self.y.loc[neighbors]
            result.append(np.mean(y))
        return result
    
    def predict(self, X):
        idx_test = 0
        idx_train = 0
        self.dist_matrix = [[float('inf') for i in range(self.X.shape[0])] for j in range(X.shape[0])]
        for _, row_test in X.iterrows():
            for _, row_train in self.X.iterrows():
                dist = self.calc_metric[self.metric](row_test.values, row_train.values)
                self.dist_matrix[idx_test][idx_train] = (dist, idx_train)
                idx_train += 1
            idx_test += 1
            idx_train = 0
        neighbors_list = []
        for row in self.dist_matrix:
            neighbors = sorted(row, key=lambda x: x[0])[:self.k]
            neighbors_idxs = [idx for dist, idx in neighbors]
            neighbors_list.append(neighbors_idxs)
        target = self.calc_distance[self.weight](neighbors_list)
        return np.array(target)
