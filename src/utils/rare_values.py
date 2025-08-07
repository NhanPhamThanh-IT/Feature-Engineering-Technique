import pandas as pd

class GroupingRareValues:
    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        self._dim = X.shape[1]
        _, categories = self.grouping(X, self.threshold, self.mapping, self.cols)
        self.mapping = categories
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError("Must train encoder before transform.")
        if X.shape[1] != self._dim:
            raise ValueError(f"Unexpected input dimension {X.shape[1]}, expected {self._dim}")
        X, _ = self.grouping(X, self.threshold, self.mapping, self.cols)
        return X

    def grouping(self, X, threshold, mapping=None, cols=None):
        X = X.copy()
        if mapping is not None:
            mapping_out = mapping
            for m in mapping:
                col = m["col"]
                X[col] = X[col].map(m["mapping"])
        else:
            mapping_out = []
            for col in cols:
                freq = X[col].value_counts(normalize=True)
                mapping_dict = {k: (k if k in freq[freq >= threshold].index else "rare") for k in freq.index}
                mapping_series = pd.Series(mapping_dict)
                mapping_out.append({"col": col, "mapping": mapping_series, "data_type": X[col].dtype})
        return X, mapping_out


class ModeImputation:
    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        self._dim = X.shape[1]
        _, categories = self.impute_with_mode(X, self.threshold, self.mapping, self.cols)
        self.mapping = categories
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError("Must train encoder before transform.")
        if X.shape[1] != self._dim:
            raise ValueError(f"Unexpected input dimension {X.shape[1]}, expected {self._dim}")
        X, _ = self.impute_with_mode(X, self.threshold, self.mapping, self.cols)
        return X

    def impute_with_mode(self, X, threshold, mapping=None, cols=None):
        X = X.copy()
        if mapping is not None:
            mapping_out = mapping
            for m in mapping:
                col = m["col"]
                X[col] = X[col].map(m["mapping"])
        else:
            mapping_out = []
            for col in cols:
                freq = X[col].value_counts(normalize=True)
                mode = X[col].mode()[0]
                mapping_dict = {k: (k if k in freq[freq >= threshold].index else mode) for k in freq.index}
                mapping_series = pd.Series(mapping_dict)
                mapping_out.append({"col": col, "mapping": mapping_series, "data_type": X[col].dtype})
        return X, mapping_out
