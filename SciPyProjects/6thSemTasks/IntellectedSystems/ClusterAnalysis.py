import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype, int64, float64
from pandas import DataFrame, Int64Dtype
from scipy.cluster._hierarchy import linkage
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize, StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


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

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(data)
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.show()
