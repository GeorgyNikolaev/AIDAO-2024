from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
import pandas as pd
import umap
import numpy as np


# Размерность временного ряда после сжатия.
n_components_1 = 175
n_components_2 = 150

# Загрузка данных.
data = np.load('ihb.npy')
data = np.nan_to_num(data)

# Стандартизация и нормализация данных.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(320, 10 * 246))
norma = Normalizer()
data_norm = norma.fit_transform(data_scaled).reshape(320 * 10, 246)

# Применяем PCA и UMAP для уменьшения размерности.
pca = PCA(n_components=n_components_1)
reduced_data = pca.fit_transform(data_norm)

umap_model = umap.UMAP(n_components=n_components_2)
reduced_data = np.array(umap_model.fit_transform(reduced_data)).reshape((320, 10, n_components_2))

# Матрицы корреляции для каждого объекта.
corr_matrix = []

for i in range(320):
    corr = np.corrcoef(reduced_data[i].T)       # Расчет корреляционной матрицы для параметров.
    corr = (corr + corr.T) / 2      # Делание матрицы симметричной, так как могут быть неточности в вычислениях.
    np.fill_diagonal(corr, 1)       # Заполнение единицами главной диагонали, так как могут быть неточности.
    corr = np.nan_to_num(corr)      # Заполняем все NaN нулями.
    corr = 1 - np.abs(corr)     # Делаем так, что матрица показывала насколько параметры "не схожи".
    corr = squareform(corr)     # Подсчет расстояния между каждыми точками матрицы корреляции.
    corr_matrix.append(corr)

n = 3
# Сжатие данных до [320, 3].
tsne = TSNE(n_components=n)
reduced_data = np.array(tsne.fit_transform(np.array(corr_matrix))).reshape((320, n))

# Кластеризация.
model = KMeans(n_clusters=20)
model.fit(reduced_data)

cluster_distances = model.transform(reduced_data)

# Разбиение данных ровно по 16 объектов на каждый кластер.
labeling = np.zeros(len(data), dtype=int)
leftover_indexes = np.arange(len(data))
for i in range(20):
    distances_from_current_cluster_center = cluster_distances[:, i]
    if len(distances_from_current_cluster_center) > 16:
        top16 = np.argpartition(distances_from_current_cluster_center, 16)[:16]
        labeling[leftover_indexes[top16]] = i
        cluster_distances = np.delete(cluster_distances, top16, axis=0)
        leftover_indexes = np.delete(leftover_indexes, top16)
    else:
        labeling[leftover_indexes] = i

# Сохранение результата.
pd.DataFrame({'prediction': labeling}).to_csv('submission.csv', index=False)



