import numpy as np


class Node:
    def __init__(self, question=None, true_branch=None, false_branch=None, uncertainty=None, *, leaf_value=None):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.uncertainty = uncertainty
        self.leaf_value = leaf_value

    @property
    def _is_leaf_node(self):
        return self.leaf_value is not None


class Question:
    def __init__(self, column_index, value, header):
        self.column_index = column_index
        self.value = value
        self.header = header

    def match(self, example):
        if isinstance(example, list):
            example = np.array(example, dtype="O")
        val = example[self.column_index]

        # adding numpy int and float data types as well
        if isinstance(val, (int, float, np.int64, np.float64)):
            # a condition for question to return True or False for numeric value
            return float(val) >= float(self.value)
        else:
            return str(val) == str(self.value)  # categorical data comparison

    def __repr__(self):
        condition = "=="
        if isinstance(self.value, (int, float, np.int64, np.float64)):
            condition = ">="
        return f"Is {self.header} {condition} {self.value} ?"


class DecisionTreeClassifier:
    def __init__(self, max_depth=100, min_samples_split=2, criteria='gini'):
        self._X = None
        self._y = None
        self._feature_names = None
        self._target_name = None
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criteria = criteria

    def _count_dict(self, a):
        unique_values = np.unique(a, return_counts=True)
        zipped = zip(*unique_values)
        return dict(zipped)

    def _gini_impurity(self, arr):
        classes, counts = np.unique(arr, return_counts=True)
        gini_score = 1 - np.square(counts / arr.shape[0]).sum(axis=0)
        return gini_score

    def _entropy(self, arr):
        classes, counts = np.unique(arr, return_counts=True)
        p = counts / arr.shape[0]
        entropy_score = (-p * np.log2(p)).sum(axis=0)
        return entropy_score

    def _partition(self, rows, question):
        true_idx, false_idx = [], []
        for idx, row in enumerate(rows):
            if question.match(row):
                true_idx.append(idx)
            else:
                false_idx.append(idx)
        return true_idx, false_idx

    def _info_gain(self, left, right, parent_uncertainty):

        pr = left.shape[0] / (left.shape[0] + right.shape[0])

        if self.criteria == "entropy":
            child_uncertainty = pr * \
                self._entropy(left) - (1 - pr) * self._entropy(right)
        else:
            child_uncertainty = pr * \
                self._gini_impurity(left) - (1 - pr) * self._gini_impurity(right)

        info_gain_value = parent_uncertainty - child_uncertainty
        return info_gain_value

    def _find_best_split(self, X, y):

        max_gain = -1
        best_split_question = None

        if self.criteria == "entropy":
            parent_uncertainty = self._entropy(y)
        else:
            parent_uncertainty = self._gini_impurity(y)

        m_samples, n_labels = X.shape

        for col_index in range(n_labels):  # iterate over feature columns
            # get unique values from the feature
            unique_values = np.unique(X[:, col_index])
            for val in unique_values:  # check for every value and find maximum info gain

                ques = Question(
                    column_index=col_index,
                    value=val,
                    header=self._feature_names[col_index]
                )

                t_idx, f_idx = self._partition(X, ques)
                # if it does not split the data
                # skip it
                if len(t_idx) == 0 or len(f_idx) == 0:
                    continue

                true_y = y[t_idx, :]
                false_y = y[f_idx, :]

                gain = self._info_gain(true_y, false_y, parent_uncertainty)
                if gain > max_gain:
                    max_gain, best_split_question = gain, ques

        return max_gain, best_split_question, parent_uncertainty

    def _build_tree(self, X, y, depth=0):
        m_samples, n_labels = X.shape
        if (depth > self.max_depth or n_labels == 1 or m_samples < self.min_samples_split):
            return Node(leaf_value=self._count_dict(y))

        gain, ques, uncertainty = self._find_best_split(X, y)

        if gain == 0:
            return Node(leaf_value=self._count_dict(y))

        t_idx, f_idx = self._partition(X, ques)
        true_branch = self._build_tree(X[t_idx, :], y[t_idx, :], depth + 1)
        false_branch = self._build_tree(X[f_idx, :], y[f_idx, :], depth + 1)
        return Node(
            question=ques,
            true_branch=true_branch,
            false_branch=false_branch,
            uncertainty=uncertainty
        )

    def train(self, X, y, feature_name=None, target_name=None):

        X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X
        y = np.array(y, dtype='O') if not isinstance(y, (np.ndarray)) else y

        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        self._feature_names = feature_name or [f"C_{i}" for i in range(self._X.shape[1])]
        self._target_name = target_name or ['target']

        self._tree = self._build_tree(
            X=self._X,
            y=self._y
        )

    def print_tree(self, node=None, spacing="|-"):

        node = node or self._tree

        if node._is_leaf_node:
            print(spacing, " Predict :", node.leaf_value)
            return

        # Print the question at this node
        print(spacing + str(node.question) +
              " | "+ self.criteria +" :" + str(node.uncertainty))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, "  " + spacing + "-")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, "  " + spacing + "-")

    def _classification(self, row, node):

        if node._is_leaf_node:
            return node.leaf_value

        if node.question.match(row):
            return self._classification(row, node.true_branch)
        else:
            return self._classification(row, node.false_branch)

    def _print_leaf_probability(self, results):
        total = sum(results.values())
        probs = {}
        for key in results:
            probs[key] = (results[key] / total) * 100
        return probs

    def predict(self, X):
        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                return self._classification(row=X, node=self._tree)
            else:
                leaf_value = []
                for row in X:
                    max_result = 0
                    result_dict = self._classification(row=row, node=self._tree)
                    result = None
                    for key in result_dict:
                        if result_dict[key] > max_result:
                            result = key                           
                    leaf_value.append([result])
                return np.array(leaf_value, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")

    def predict_probability(self, X):

        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                return self._print_leaf_probability(self._classification(row=X, node=self._tree))
            else:
                leaf_value = []
                for row in X:
                    leaf_value.append([self._print_leaf_probability(
                        self._classification(row=row, node=self._tree))])
                return np.array(leaf_value, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=3, criteria='variance'):
        self._X = None
        self._y = None
        self._feature_names = None
        self._target_name = None
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criteria = criteria

    def _mean_leaf_value(self, a):
        return np.mean(a)

    def _partition(self, rows, question):
        true_idx, false_idx = [], []
        for idx, row in enumerate(rows):
            if question.match(row):
                true_idx.append(idx)
            else:
                false_idx.append(idx)
        return true_idx, false_idx

    def _info_gain(self, left, right, parent_uncertainty):

        pr = left.shape[0] / (left.shape[0] + right.shape[0])
        
        if self.criteria == "variance":
            child_uncertainty = pr * np.var(left) - (1 - pr) * np.var(right)
        else:
            raise ValueError(f"{self.criteria} is not available. Try variance.")
        
        info_gain_value = parent_uncertainty - child_uncertainty
        return info_gain_value

    def _find_best_split(self, X, y):

        max_gain = -1
        best_split_question = None
        
        if self.criteria == "variance":
            parent_uncertainty = np.var(y)
        else:
            raise ValueError(f"{self.criteria} is not available. Try variance.")
            

        m_samples, n_labels = X.shape

        for col_index in range(n_labels):  # iterate over feature columns
            # get unique values from the feature
            unique_values = np.unique(X[:, col_index])
            for val in unique_values:  # check for every value and find maximum info gain

                ques = Question(
                    column_index=col_index,
                    value=val,
                    header=self._feature_names[col_index]
                )

                t_idx, f_idx = self._partition(X, ques)
                # if it does not split the data
                # skip it
                if len(t_idx) == 0 or len(f_idx) == 0:
                    continue

                true_y = y[t_idx, :]
                false_y = y[f_idx, :]

                gain = self._info_gain(true_y, false_y, parent_uncertainty)
                if gain > max_gain:
                    max_gain, best_split_question = gain, ques

        return max_gain, best_split_question, parent_uncertainty

    def _build_tree(self, X, y, depth=0):
        m_samples, n_labels = X.shape
        if (depth > self.max_depth or n_labels == 1 or m_samples < self.min_samples_split):
            return Node(leaf_value=self._mean_leaf_value(y))

        gain, ques, uncertainty = self._find_best_split(X, y)

        if gain == 0:
            return Node(leaf_value=self._mean_leaf_value(y))

        t_idx, f_idx = self._partition(X, ques)
        true_branch = self._build_tree(X[t_idx, :], y[t_idx, :], depth + 1)
        false_branch = self._build_tree(X[f_idx, :], y[f_idx, :], depth + 1)
        return Node(
            question=ques,
            true_branch=true_branch,
            false_branch=false_branch,
            uncertainty=uncertainty
        )

    def train(self, X, y, feature_name=None, target_name=None):

        X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X
        y = np.array(y, dtype='O') if not isinstance(y, (np.ndarray)) else y

        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        self._feature_names = feature_name or [f"C_{i}" for i in range(self._X.shape[1])]
        self._target_name = target_name or ['target']

        self._tree = self._build_tree(
            X=self._X,
            y=self._y
        )

    def print_tree(self, node=None, spacing="|-"):

        node = node or self._tree

        if node._is_leaf_node:
            print(spacing, " Predict :", node.leaf_value)
            return

        # Print the question at this node
        print(spacing + str(node.question) +
              " | "+ self.criteria +" :" + str(node.uncertainty))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, "  " + spacing + "-")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, "  " + spacing + "-")

    def _regression(self, row, node):

        if node._is_leaf_node:
            return node.leaf_value

        if node.question.match(row):
            return self._regression(row, node.true_branch)
        else:
            return self._regression(row, node.false_branch)


    def predict(self, X):
        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                return self._regression(row=X, node=self._tree)
            else:
                leaf_value = []
                for row in X:
                    result = self._regression(row=row, node=self._tree)                     
                    leaf_value.append([result])
                return np.array(leaf_value, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")

