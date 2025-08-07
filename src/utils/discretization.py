import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class ChiMerge():
    def __init__(self, col=None, bins=None, confidenceVal=3.841, num_of_bins=10):
        self.col = col
        self._dim = None
        self.confidenceVal = confidenceVal
        self.bins = bins
        self.num_of_bins = num_of_bins

    def fit(self, X, y, **kwargs):
        self._dim = X.shape[1]
        _, bins = self.chimerge(
            X_in=X,
            y=y,
            confidenceVal=self.confidenceVal,
            col=self.col,
            num_of_bins=self.num_of_bins
        )
        self.bins = bins
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))
        X, _ = self.chimerge(
            X_in=X,
            col=self.col,
            bins=self.bins
        )
        return X

    def chimerge(self, X_in, y=None, confidenceVal=None, num_of_bins=None, col=None, bins=None):
        X = X_in.copy(deep=True)
        if bins is not None:
            try:
                X[col+'_chimerge'] = pd.cut(X[col], bins=bins, include_lowest=True)
            except Exception as e:
                print(e)
        else:
            try:
                total_num = X.groupby([col])[y].count()
                total_num = pd.DataFrame({'total_num': total_num}) 
                positive_class = X.groupby([col])[y].sum()
                positive_class = pd.DataFrame({'positive_class': positive_class}) 
                regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True, how='inner')  
                regroup.reset_index(inplace=True)
                regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  
                regroup = regroup.drop('total_num', axis=1)
                np_regroup = np.array(regroup)  

                i = 0
                while (i <= np_regroup.shape[0] - 2):
                    if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                        np_regroup[i, 1] += np_regroup[i + 1, 1]
                        np_regroup[i, 2] += np_regroup[i + 1, 2]
                        np_regroup[i, 0] = np_regroup[i + 1, 0]
                        np_regroup = np.delete(np_regroup, i + 1, 0)
                        i -= 1
                    i += 1

                chi_table = np.array([])
                for i in np.arange(np_regroup.shape[0] - 1):
                    chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
                          * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
                          ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * 
                           (np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
                    chi_table = np.append(chi_table, chi)

                while True:
                    if (len(chi_table) <= (num_of_bins - 1) and min(chi_table) >= confidenceVal):
                        break
                    chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  
                    np_regroup[chi_min_index, 1] += np_regroup[chi_min_index + 1, 1]
                    np_regroup[chi_min_index, 2] += np_regroup[chi_min_index + 1, 2]
                    np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
                    np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

                    if (chi_min_index == np_regroup.shape[0] - 1): 
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - 
                                                        np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + 
                                                          np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * 
                                                        (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * 
                                                        (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * 
                                                        (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table = np.delete(chi_table, chi_min_index, axis=0)
                    else:
                        chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - 
                                                        np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + 
                                                          np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * 
                                                        (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * 
                                                        (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * 
                                                        (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                        chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - 
                                                    np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                                   * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + 
                                                      np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                                   ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * 
                                                    (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * 
                                                    (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * 
                                                    (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                        chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)

                result_data = pd.DataFrame()
                result_data['variable'] = [col] * np_regroup.shape[0]
                bins = []
                tmp = []
                for i in np.arange(np_regroup.shape[0]):
                    if i == 0:
                        y = '-inf' + ',' + str(np_regroup[i, 0])
                    elif i == np_regroup.shape[0] - 1:
                        y = str(np_regroup[i - 1, 0]) + '+'
                    else:
                        y = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
                    bins.append(np_regroup[i - 1, 0])
                    tmp.append(y)

                bins.append(X[col].min() - 0.1)
                result_data['interval'] = tmp  
                result_data['flag_0'] = np_regroup[:, 2] 
                result_data['flag_1'] = np_regroup[:, 1]  
                bins.sort(reverse=False)
                print('Interval for variable %s' % col)
                print(result_data)

            except Exception as e:
                print(e)

        return X, bins


class DiscretizeByDecisionTree:
    def __init__(self, col=None, max_depth=None, tree_model=None):
        self.col = col
        self.max_depth = max_depth
        self.tree_model = tree_model
        self._dim = None

    def fit(self, X, y):
        self._dim = X.shape[1]
        _, self.tree_model = self._discretize(X.copy(), y)
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError("Call 'fit' before 'transform'.")
        if X.shape[1] != self._dim:
            raise ValueError(f"Unexpected input dimension {X.shape[1]}, expected {self._dim}.")

        X, _ = self._discretize(X.copy(), tree_model=self.tree_model)
        return X

    def _discretize(self, X, y=None, max_depth=None, tree_model=None):
        col = self.col

        if tree_model:
            X[f'{col}_tree_discret'] = tree_model.predict_proba(X[[col]])[:, 1]
            return X, tree_model

        if isinstance(self.max_depth, int):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
        elif isinstance(self.max_depth, list):
            scores = []
            for depth in self.max_depth:
                model = DecisionTreeClassifier(max_depth=depth)
                score = cross_val_score(model, X[[col]], y, cv=3, scoring='roc_auc').mean()
                scores.append((depth, score))
            best_depth = max(scores, key=lambda x: x[1])[0]
            tree = DecisionTreeClassifier(max_depth=best_depth)
        else:
            raise ValueError("max_depth must be int or list of int")

        tree.fit(X[[col]], y)
        X[f'{col}_tree_discret'] = tree.predict_proba(X[[col]])[:, 1]
        return X, tree
