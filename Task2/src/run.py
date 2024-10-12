import pickle

import numpy as np
import pandas as pd
from scripts.data_utils import get_connectome

# Загрузка данных.
X = np.load('./data/ts_cut/HCPex/predict.npy')
# Данные имеют разную длину, поэтому приводим их к виду 419x419.
X = get_connectome(X)

# Приведение к нужному типу
X = X.reshape(len(X), len(X[0]) * len(X[0][0]))

# Загрузка модели.
model_file_name = 'model.pkl'
with open(model_file_name, 'rb') as file:
    model = pickle.load(file)

# Тест модели
y_pred = model.predict(X).argmax(axis=1)

# Сохранение результата
solution = pd.DataFrame(data=y_pred, columns=['prediction'])
solution.to_csv('./solution.csv', index=False)