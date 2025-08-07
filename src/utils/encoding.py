import pandas as pd

class MeanEncoding:
    def __init__(self, mapping=None, cols=None):
        self.cols = cols
        self.mapping = mapping
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        self._dim = X.shape[1]
        _, categories = self.mean_encoding(X, y, mapping=self.mapping, cols=self.cols)
        self.mapping = categories
        return self

    def transform(self, X):
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')
        if X.shape[1] != self._dim:
            raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')
        X, _ = self.mean_encoding(X, mapping=self.mapping, cols=self.cols)
        return X

    def mean_encoding(self, X_in, y=None, mapping=None, cols=None):
        X = X_in.copy(deep=True)
        if mapping is not None:
            mapping_out = mapping
            for i in mapping:
                column = i.get('col')
                X[column] = X[column].map(i['mapping'])
        else:
            mapping_out = []
            for col in cols:
                mapping = X[y.name].groupby(X[col]).mean().to_dict()
                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype})
        return X, mapping_out
