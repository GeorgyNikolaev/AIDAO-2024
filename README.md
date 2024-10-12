# AIDAO 2024
## Task 1

---

В первом задании нужно было кластеризовать неразмеченные данные. 
Были данные ФМРТ 20 людей по 16 наблюдений. Каждое наблюдение это 10 времеянных рядов. А в каждом ряде
по 246 параметров. Таким образов датасет имел размерность [20 * 17, 10, 246].
Данные имели особенную специфику, поэтому я выбрал следующую стратегию:

1. Стандартизация и нормализация данных.
2. Сокращение размерности с помощью PCA и UMAP алгоритмов.
3. Выделение корреляционных матриц для каждого объекта.
4. Замена значений на ```"1 - corr_value" ``` чтобы иметь не *схожесть*, а *расхожесть* данных.
5. Нахождения расстояний для каждого ```corr_value```.
6. Сжатие с помощью TSNE алгоритма.
7. Кластеризация обычным KMeans.
8. Делаем так, чтобы в каждом кластере было ровно по 16 наблюдений.

Это все, что потребовалось для результата в 0.28. Конечно это далеко не лучшая 
стратегия. Я пробовал большое множество решений, но все они давали либо такие же,
либо хуже результаты.

## Task 2

---

Во втором задании нужно было классифицировать состояние человека, а именно спит он или нет. Задание я сделал за 10-15 минут на основе библиотеки ```keras```, 
впоследствии я переписал его на ```sklearn``` поскольку у компилятора Яндекса
не подгружена ```tensorflow```. Моя достаточно тривиальная модель 
с одним полно связным слоем в 32 нейрона дала 0.825 score. Поэтому я сразу вернулся 
к первому заданию, так как там был скверный score. Я пробовал больше слоев,
но ограничение по памяти не давало мне это сделать. Файл с весами слишком 
много весил. Перейдем к стратегии:

1. Рассчитываем матрицы корреляции для данных.
2. Преобразуем ```y_train``` и ```y_test``` в *one-hot* вектора.
3. Создаем модель с одним полно связным слоем в 32 нейрона с оптимизатором ```Adam```, активацией ```relu``` и шагом сходимости 0.095.
4. Обучаем данные и выводим *accuracy*.

Конечно это самая банальная стратегия, но она выдала 0.825 score. 
Так же мне нужно было улучшать первую задачу, поэтому впоследствии я особо не пытался 
менять стратегию.
