import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype, int64, float64
from pandas import DataFrame, Int64Dtype
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc


def row_to_intenum(row: [str]) -> [int]:
    row_unique_values = list(set(row))
    mapping = {key: value for value, key in enumerate(row_unique_values)}

    return list(map(lambda item: mapping[item], row))


def optimize_csv(csv: DataFrame):
    for column_alias in csv:
        current_column = csv[column_alias]

        if current_column.dtypes is np.dtype(object):
            csv[column_alias] = normalize(np.array(row_to_intenum(current_column))[:, np.newaxis], axis=0).ravel()

        elif current_column.dtypes is np.dtype(int64) or current_column.dtypes is np.dtype(float64):
            csv[column_alias] = normalize(np.array(current_column)[:, np.newaxis], axis=0).ravel()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    data = pd.read_csv('res/adult.data')
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education num',
                    'marital status', 'occupation', 'relationship', 'raсe', 'sex',
                    '+capital', '-capital', 'work/hrs', 'native country', 'state']
    optimize_csv(data)

    del data['native country']
    del data['raсe']
    del data['education']

    data['work/hrs'] = preprocessing.scale(data['work/hrs'])

    k_means = KMeans()
    k_means.fit(data)

    embedding = MDS(n_components=2)
    x = embedding.fit_transform(k_means.cluster_centers_)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()