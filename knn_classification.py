class MyKNNClf:
    def __init__(self, k=3, metric="euclidean", weight="uniform"):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.calc_distance = {
            "euclidean": self.calc_euclidean,
            "chebyshev": self.calc_chebyshev,
            "manhattan": self.calc_manhattan,
            "cosine": self.calc_cosine,
        }
        self.weight = weight
        self.get_classes_probas = {
            "uniform": self.calc_uniform,
            "rank": self.calc_rank_weights,
            "distance": self.calc_distance_weights,
        }

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])

    def calc_euclidean(self, arr1, arr2):
        return np.sqrt(np.sum(np.square(arr1 - arr2)))

    def calc_chebyshev(self, arr1, arr2):
        return np.max(np.abs(arr1 - arr2))

    def calc_manhattan(self, arr1, arr2):
        return np.sum(np.abs(arr1 - arr2))

    def calc_cosine(self, arr1, arr2):
        return 1 - (np.sum([x * y for x, y in zip(arr1, arr2)])) / np.sqrt(
            np.sum(np.square(arr1)) * np.sum(np.square(arr2))
        )

    def get_nearest_neighbors(self, X_test):
        self.distance_matrix = [
            [float("inf") for i in range(self.X.shape[0])]
            for j in range(X_test.shape[0])
        ]
        test_idx = 0
        train_idx = 0
        for _, test_row in X_test.iterrows():
            for _, train_row in self.X.iterrows():
                self.distance_matrix[test_idx][train_idx] = self.calc_distance[
                    self.metric
                ](test_row, train_row)
                train_idx += 1
            test_idx += 1
            train_idx = 0
        nearest_neighbors = [
            sorted(zip(row, self.y), key=lambda x: x[0])[: self.k]
            for row in self.distance_matrix
        ]
        knn_classes_probas = self.get_classes_probas[self.weight](nearest_neighbors)
        return knn_classes_probas

    def calc_rank_weights(self, nearest_neighbors):
        result = []
        for row in nearest_neighbors:
            distance = [dist for dist, y in row]
            target = np.array([y for distance, y in row])
            first_class_weight = np.sum(
                1 / (np.argwhere(target == 1).squeeze() + 1)
            ) / np.sum(1 / np.arange(1, len(target) + 1))
            second_class_weight = np.sum(
                1 / (np.argwhere(target == 0).squeeze() + 1)
            ) / np.sum(1 / np.arange(1, len(target) + 1))
            predicted_class = 1 if first_class_weight > second_class_weight else 0
            predicted_proba = first_class_weight
            result.append((predicted_class, predicted_proba))
        return result

    def calc_distance_weights(self, nearest_neighbors):
        result = []
        for row in nearest_neighbors:
            distance = np.array([dist for dist, y in row])
            target = np.array([y for distance, y in row])
            first_class_weight = np.sum(
                1 / distance[np.argwhere(target == 1)].squeeze()
            ) / np.sum(1 / distance)
            second_class_weight = np.sum(
                1 / distance[np.argwhere(target == 0)].squeeze()
            ) / np.sum(1 / distance)

            predicted_class = 1 if first_class_weight > second_class_weight else 0
            predicted_proba = first_class_weight
            result.append((predicted_class, predicted_proba))
        return result

    def calc_uniform(self, nearest_neighbors):
        result = []
        for row in nearest_neighbors:
            target = [y for distance, y in row]
            predicted_class = (
                1 if np.sum(target) >= (len(target) - np.sum(target)) else 0
            )
            predicted_proba = np.sum(target) / len(target)
            result.append((predicted_class, predicted_proba))
        return result

    def predict(self, X_test):
        knn_classes_probas = self.get_nearest_neighbors(X_test)
        classes = [predicted_class for predicted_class, proba in knn_classes_probas]
        return np.array(classes)

    def predict_proba(self, X_test):
        knn_classes_probas = self.get_nearest_neighbors(X_test)
        probas = [proba for predicted_class, proba in knn_classes_probas]
        return np.array(probas)

    def __repr__(self):
        return f"MyKNNClf class: k={self.k}"
