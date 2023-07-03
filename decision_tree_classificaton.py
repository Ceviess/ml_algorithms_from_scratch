class MyTreeClf():
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.leafs_cnt = 1
        self.bins = bins
        self.criterion = criterion
        self.calc_criterion = {
            'entropy': self.calc_entropy,
            'gini': self.calc_gini
        }
        
    def __repr__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def calc_gini(self, arr):
        if len(arr) == 0 or len(set(arr)) == 1:
            return 0
        first_class = sum(arr) / len(arr)
        second_class = 1 - first_class
        return 1 - (np.square(first_class) + np.square(second_class))   
    
    def calc_class(self, class_count, n):
        return (class_count / n) * np.log2(class_count / n) if class_count != 0 else 0
    
    def calc_entropy(self, arr):
        n = arr.shape[0]
        pos = np.sum(arr)
        neg = n - np.sum(arr)

        entropy = -(self.calc_class(pos, n) + self.calc_class(neg, n))
        return entropy

    def get_best_split(self, X, y):
        def calc_gain(left, right, overall_criterion):
            left_criterion = self.calc_criterion[self.criterion](left)
            right_criterion = self.calc_criterion[self.criterion](right)
            
            n = left.shape[0] + right.shape[0]
            left_weight = left.shape[0] / n
            right_weight = right.shape[0] / n

            gain = overall_criterion - (left_weight * left_criterion + right_weight * right_criterion)
            return gain
        
        def get_thresholds(feat):
            unique_values = feat[feat.columns[0]].unique()
            if not self.bins:
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(unique_values.shape[0] - 1)]
            else:
                thresholds = self.thresholds[feat.columns[0]]
            return thresholds

        def calc_feat(feat):
            best_gain = -float('inf')
            best_threshold = None
            overall_criterion = self.calc_criterion[self.criterion](feat['y'].values)

            thresholds = get_thresholds(feat)
            for threshold in thresholds: 
                left = feat.loc[feat[feat.columns[0]] <= threshold, 'y'].values
                right = feat.loc[feat[feat.columns[0]] > threshold, 'y'].values
                gain = calc_gain(left, right, overall_criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
            return best_gain, best_threshold

        df = X.copy()
        df['y'] = y
        best_col = None
        best_split_value = None
        best_gain = -float('inf')

        for col in X.columns:
            feat = df[[col, 'y']].sort_values(col)
            cur_gain, cur_threshold = calc_feat(feat)
            if cur_gain > best_gain:
                best_col = col
                best_split_value = cur_threshold
                best_gain = cur_gain
        return best_col, best_split_value, best_gain
    
    def check_leaf(self, X, y, depth):
        if len(set(y)) == 1:
            return True
        if depth >= self.max_depth:
            return True
        if X.shape[0] < self.min_samples_split:
            return True
        if self.leafs_cnt >= self.max_leafs:
            return True
    
    def fit(self, X, y):
        self.n_rows = X.shape[0]
        self.fi = {}
        self.thresholds = {}
        if self.bins:
            for col in X.columns:
                unique_values = X[col].sort_values().unique()
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(unique_values.shape[0] - 1)]
                if len(thresholds) > (self.bins - 1):
                    _, thresholds = np.histogram(unique_values, bins=self.bins)
                    thresholds = thresholds[1:-1]
                self.thresholds[col] = thresholds
        for col in X.columns:
            self.fi[col] = 0
        self.tree = self.build_tree(X, y)
        self.update_feat_importance()
        
    def update_feat_importance(self):
            def walk_tree(node):
                if node['leaf']:
                    return
                self.fi[node['feature']] += node['feat_importance']
                walk_tree(node['left_child'])
                walk_tree(node['right_child'])
                
            walk_tree(self.tree)
            
    def calc_feat_importance(self, cur_node_rows, y, left_mask, right_mask):
        cur_criterion = self.calc_criterion[self.criterion](y)
        left_criterion = self.calc_criterion[self.criterion](y[left_mask])
        right_criterion = self.calc_criterion[self.criterion](y[right_mask])
        
        left_weight = y[left_mask].shape[0] / y.shape[0]
        right_weight = y[right_mask].shape[0] / y.shape[0]
        feat_importance = cur_node_rows / self.n_rows * (cur_criterion - left_weight * left_criterion - right_weight * right_criterion)
        return feat_importance

    def build_tree(self, X, y, depth=0):
        if self.check_leaf(X, y, depth):
            return {"leaf": True, "class": np.sum(y) / y.shape[0], "length": len(y)}
        self.leafs_cnt += 1
        best_col, best_split_value, best_ig = self.get_best_split(X, y)
        left_mask = X[best_col] <= best_split_value
        right_mask = X[best_col] > best_split_value
        left_child = self.build_tree(X[left_mask], y[left_mask], depth=depth+1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth=depth+1)
        cur_node_rows = X.shape[0]
        feat_importance = self.calc_feat_importance(cur_node_rows, y, left_mask, right_mask)
        node = {
            "leaf": False,
            "feature": best_col,
            "split_value": best_split_value,
            "ig": best_ig,
            "left_child": left_child,
            "right_child": right_child,
            "feat_importance": feat_importance
        }
        return node
    
    def predict(self, X):
        def find_node(X, node):
            if node['leaf']:
                return node['class']
            if X[node['feature']] <= node['split_value']:
                return find_node(X, node['left_child'])
            else:
                return find_node(X, node['right_child'])
        
        result = []
        for idx, row in X.iterrows():
            result.append(find_node(row, self.tree))
            
        return np.array(result)
    
    def print_tree(self):
        def print_node(node, indent=""):
            if node['leaf']:
                print(indent + "leaf = " + str(node['class']))
            else:
                print(indent + f'{node["feature"]} <= {node["split_value"]}')
                print_node(node['left_child'], indent+'\t')
                print_node(node['right_child'], indent+'\t')
        if self.tree:
            print_node(self.tree)
