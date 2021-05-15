from typing import Union
import numpy as np
from __future__ import annotations


class Question:
    def __init__(self, column_index: int, value: Union[int, float, str], header: str):
        self.column_index = column_index
        self.value = value
        self.header = header

    def match(self, example: Union[list, np.ndarray]):
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


class LeafNode:
    def __init__(self, data: Union[np.ndarray,list]):
        unique_values = np.unique(np.array(data, dtype='O')[:, -1], return_counts=True)
        zipped = zip(*unique_values)
        self.predictions = dict(zipped)


class DecisionNode:
    def __init__(self, question: Question, true_branch: DecisionNode, false_branch: DecisionNode, inconsistency: float):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.inconsistency = inconsistency


class DecisionTreeClassifier:

    def __init__(self, tree_depth=None):
        self._X = None
        self._y = None
        self._feature_name = None
        self._target_name = None
        self._tree = None
        self.tree_depth = tree_depth
        self._depth = tree_depth

    def _gini_impurity(self, arr):
        arr = np.array(arr, dtype='O') if not isinstance(arr, (np.ndarray)) else arr
        arr = arr.reshape(-1, 1) if len(arr.shape) == 1 else arr

        sum_val = 0
        classes = np.unique(arr[:, -1])

        for cl in classes:
            p = len(np.where(arr == cl)[0]) / arr.shape[0]
            sum_val += p**2

        gini_score = 1 - sum_val
        return gini_score

    def _partition(self, rows, question):
        true_side, false_side = [], []
        for row in rows:
            if question.match(row):
                true_side.append(row)
            else:
                false_side.append(row)
        return true_side, false_side

    def _info_gain(self, left_side, right_side, current_uncertainity):

        left_side = np.array(left_side, dtype='O') if isinstance(
            left_side, (list, tuple)) else left_side
        right_side = np.array(right_side, dtype='O') if isinstance(
            right_side, (list, tuple)) else right_side

        pr = left_side.shape[0] / (left_side.shape[0] + right_side.shape[0])

        info_gain_value = current_uncertainity - pr * \
            self._gini_impurity(left_side) - (1 - pr) * self._gini_impurity(right_side)

        return info_gain_value

    def _find_best_split(self, rows, headers):

        rows = np.array(rows, dtype='O') if not isinstance(rows, np.ndarray) else rows

        max_gain = 0
        best_split_question = None

        current_inconsistency = self._gini_impurity(rows)
        n = rows.shape[1] - 1

        for col_index in range(n):  # iterate over feature columns

            # get unique values from the feature
            unique_values = np.unique(rows[:, col_index])
            for val in unique_values:  # check for every value and find maximum info gain

                ques = Question(column_index=col_index, value=val,
                                header=headers[col_index])

                true_side, false_side = self._partition(rows, ques)

                # if it does not split the data
                # skip it
                if len(true_side) == 0 or len(false_side) == 0:
                    continue

                gain = self._info_gain(true_side, false_side, current_inconsistency)

                if gain > max_gain:
                    max_gain, best_split_question = gain, ques
        return max_gain, best_split_question, current_inconsistency

    def _build_tree(self, rows, headers, depth=0):
        depth += 1
        rows = np.array(rows, dtype='O') if not isinstance(rows, np.ndarray) else rows

        if self.tree_depth is not None:
            if depth > self.tree_depth:
                return LeafNode(rows)

        gain, ques, inconsistency = self._find_best_split(rows, headers)
        if gain == 0:
            return LeafNode(rows)
        true_rows, false_rows = self._partition(rows, ques)
        true_branch = self._build_tree(true_rows, headers, depth)
        false_branch = self._build_tree(false_rows, headers, depth)

        return DecisionNode(ques, true_branch, false_branch, inconsistency)

    def train(self, X, y, feature_name=None, target_name=None):

        X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X
        y = np.array(y, dtype='O') if not isinstance(y, (np.ndarray)) else y

        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        self._feature_name = feature_name or [f"C_{i}" for i in range(self._X.shape[1])]
        self._target_name = target_name or ['target']

        self._tree = self._build_tree(
            rows=np.hstack((self._X, self._y)),
            headers=self._feature_name + self._target_name
        )

    def print_tree(self, node=None, spacing="|-"):

        node = node or self._tree

        if isinstance(node, LeafNode):
            print(spacing, " Predict :", node.predictions)
            return

        # Print the question at this node
        print(spacing + str(node.question) +
              " | Inconsistency :" + str(node.inconsistency))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, "  " + spacing + "-")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, "  " + spacing + "-")

    def _classification(self, row, node):
        if isinstance(node, LeafNode):
            return node.predictions

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
                predictions = []
                for row in X:
                    predictions.append(self._classification(row=row, node=self._tree))
                return np.array(predictions, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")

    def predict_probability(self, X):

        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                return self._print_leaf_probability(self._classification(row=X, node=self._tree))
            else:
                predictions = []
                for row in X:
                    predictions.append(self._print_leaf_probability(
                        self._classification(row=row, node=self._tree)))
                return np.array(predictions, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    dataset = load_iris()

    X = dataset.data
    y = dataset.target
    feature_name = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.30)

    dt = DecisionTreeClassifier(tree_depth=3)
    dt.train(X=X_train, y=y_train, feature_name=feature_name)
    y_pred = dt.predict_probability(X_test)

    print(pd.DataFrame(np.vstack((y_pred, y_test)).T))
