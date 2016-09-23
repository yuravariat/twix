from pandas import DataFrame
from sklearn.base import TransformerMixin


class GlobalTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        # the mid 'x' value is the transformed one
        table = DataFrame(map(lambda x: int(str(x.f1).replace('V', '')), X), columns=['f1'])
        return table

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['f1']
