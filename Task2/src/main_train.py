import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scripts.data_utils import get_connectome


bnu_series_path = '../data/ts_cut/HCPex/bnu{}.npy'
bnu_labels_path = '../data/ts_cut/HCPex/bnu.csv'
ihb_series_path = '../data/ts_cut/HCPex/ihb.npy'
ihb_labels_path = '../data/ts_cut/HCPex/ihb.csv'

# Загрузка данных.
X_bnu = np.concatenate([np.load(bnu_series_path.format(i)) for i in (1, 2)], axis=0)
Y_bnu = pd.read_csv(bnu_labels_path)
X_ihb = np.load(ihb_series_path)
Y_ihb = pd.read_csv(ihb_labels_path)

# Данные имеют разную длину, поэтому приводим их к виду 419x419.
X_bnu = get_connectome(X_bnu)
X_ihb = get_connectome(X_ihb)

# Соединяем данные из разных файлов.
X = np.concatenate([X_bnu, X_ihb])
Y = np.concatenate([Y_bnu, Y_ihb])

# Разделяем данные на train и validation.
x_train, x_validate, y_train, y_validate = train_test_split(X, Y,
                                                            test_size=0.15, random_state=10)
# Приводим данные к нужному типу.
x_train = x_train.reshape(len(x_train), len(x_train[0]) * len(x_train[0][0]))
x_validate = x_validate.reshape(len(x_validate), len(x_validate[0]) * len(x_validate[0][0]))

# Предположим, x_train, y_train уже определены и y_train закодированы в формате one-hot
# Для примера, если y_train - это метки классов (0, 1), то кодируем их one-hot
n_classes = 2
y_train_onehot = np.eye(n_classes)[np.array(y_train, dtype=int)]  # Преобразование в one-hot
y_train_onehot = y_train_onehot.reshape(len(y_train_onehot), 2)

# Разделяем данные на train и validation
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train_onehot, test_size=0.1)

# Количество эпох
epochs = 10

# Создаем модель
model = MLPClassifier(hidden_layer_sizes=(32),
                      activation='relu',
                      solver='adam',
                      learning_rate_init=0.095,
                      max_iter=epochs,
                      batch_size=8)

model.fit(x_train, y_train)

# Оцениваем модель
y_validate_pred = model.predict(x_validate)
accuracy = accuracy_score(y_validate.argmax(axis=1), y_validate_pred.argmax(axis=1))  # Предполагаем, что y_validate - one-hot
print("Validation Accuracy:", accuracy)

# Сохраняем модель
pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)