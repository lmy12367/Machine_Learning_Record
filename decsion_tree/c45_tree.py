import numpy as np

class Node:
    def __init__(self):
        self.feat = None     
        self.split = None  
        self.child = []       
        self.label = None     

class DecisionTree:
    def __init__(self, lbd: float = 1.0):
        self.lbd = lbd
        self.eps = 1e-8
        self.T = 0           
        self.root = None

    def _entropy(self, y):
        cnt = np.bincount(y)
        p = cnt / cnt.sum()
        return -(p * np.log2(p + self.eps)).sum()

    def _info_gain(self, X, y, feat, val):
        H = self._entropy(y)
        left, right = y[X[:, feat] <= val], y[X[:, feat] > val]
        if len(left) == 0 or len(right) == 0:
            return 0.0
        H_left = len(left) / len(y) * self._entropy(left)
        H_right = len(right) / len(y) * self._entropy(right)
        return H - H_left - H_right

    def _split_entropy(self, X, y, feat, val):
        left, right = y[X[:, feat] <= val], y[X[:, feat] > val]
        p_l, p_r = len(left) / len(y), len(right) / len(y)
        return -(p_l * np.log2(p_l + self.eps) + p_r * np.log2(p_r + self.eps))

    def _info_gain_ratio(self, X, y, feat, val):
        ig = self._info_gain(X, y, feat, val)
        split_ent = self._split_entropy(X, y, feat, val)
        return ig / (split_ent + self.eps)

    def fit(self, X, y, feat_ranges):
        self.feat_ranges = feat_ranges
        self.feat_names = list(feat_ranges.keys())
        self.root = Node()
        self._id3(self.root, X, y)
        return self

    def _id3(self, node, X, y):
        if len(np.unique(y)) == 1:
            node.label = y[0]
            self.T += 1
            return

        best_igr, best_feat, best_val = 0, None, None
        for feat_idx in range(X.shape[1]):
            feat_name = self.feat_names[feat_idx]
            for val in self.feat_ranges[feat_name]:
                igr = self._info_gain_ratio(X, y, feat_idx, val)
                if igr > best_igr:
                    best_igr, best_feat, best_val = igr, feat_idx, val

        cur_cost = len(y) * self._entropy(y) + self.lbd
        if best_feat is None:
            new_cost = np.inf
        else:
            left_mask = X[:, best_feat] <= best_val
            right_mask = ~left_mask
            left_y, right_y = y[left_mask], y[right_mask]
            new_cost = len(left_y) * self._entropy(left_y) + \
                       len(right_y) * self._entropy(right_y) + 2 * self.lbd

        if new_cost <= cur_cost:
            node.feat, node.split = best_feat, best_val
            for mask in (left_mask, right_mask):
                child = Node()
                self._id3(child, X[mask], y[mask])
                node.child.append(child)
        else:
            node.label = np.bincount(y).argmax()
            self.T += 1

    def predict_one(self, x):
        node = self.root
        while node.label is None:
            if x[node.feat] <= node.split:
                node = node.child[0]
            else:
                node = node.child[1]
        return node.label

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()