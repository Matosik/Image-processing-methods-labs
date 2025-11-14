from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import skimage.io


def entropy(arr):
    _, counts = np.unique(arr, return_counts=True)
    counts = counts / counts.sum()
    return -(counts * np.log2(counts)).sum()


def predictor(i, j, y, number):
    if number == 1:
        if i == 0 and j == 0:
            return 0
        if j == 0:
            return y[i - 1][-1]
        return y[i][j - 1]
    if number == 2:
        if i == 0 or j == 0:
            return 0
        return int((y[i][j - 1] + y[i - 1][j]) / 2)


def MyDifCode(x, e, r):
    # Используем тип данных с большим диапазоном
    f = np.zeros(x.shape, dtype=np.int16)
    y = np.zeros(x.shape, dtype=np.int16)
    q = np.zeros(x.shape, dtype=np.int16)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            p = predictor(i, j, y, r)
            f[i][j] = x[i][j] - p
            q[i][j] = np.sign(f[i][j]) * ((abs(f[i][j]) + e) // (2 * e + 1))
            y[i][j] = p + q[i][j] * (2 * e + 1)
            # Убедимся, что разность не превышает погрешность
            assert abs(x[i][j] - y[i][j]) <= e

    return q, f


def MyDifDeCode(q, e, r):
    y = np.zeros(q.shape, dtype=np.int16)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            p = predictor(i, j, y, r)
            y[i][j] = p + q[i][j] * (2 * e + 1)

    return y


def contrasting(array):
    array_min = np.min(array)
    array_max = np.max(array)
    # Избегаем деления на ноль
    if array_max == array_min:
        return np.zeros_like(array)
    a = 255 / (array_max - array_min)
    b = (-255 * array_min) / (array_max - array_min)
    g = a * array + b
    return g.astype(np.uint8)


def main():
    # Загружаем изображение и преобразуем к int16
    img = cv.imread("test_img/01_apc.tif", cv.IMREAD_GRAYSCALE)
    img = img.astype(np.int16)  # Преобразуем к типу с большим диапазоном
    
    # Task 1
    x = range(0, 51, 5)
    y1 = [entropy(MyDifCode(img, e, 1)[0]) for e in x]
    y2 = [entropy(MyDifCode(img, e, 2)[0]) for e in x]
    plt.xlabel('Погрешность')
    plt.ylabel('Энтропия')
    plt.plot(x, y1, color="green", label="предсказатель 1")
    plt.plot(x, y2, color="red", label="предсказатель 2")
    plt.legend()
    plt.show()

    # Task 2
    e_values = [5, 10, 20, 40]  # переименовал, чтобы не конфликтовало с циклом
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(4):
        q, _ = MyDifCode(img, e_values[i], 1)
        y = MyDifDeCode(q, e_values[i], 1)
        # Преобразуем к uint8 для отображения
        y_display = np.clip(y, 0, 255).astype(np.uint8)
        axes[i].imshow(y_display, cmap="gray")
        axes[i].set_title(f"Восстановление e={e_values[i]}")
    plt.show()

    # Task 3
    e_values = [0, 0, 5, 10]  # переименовал
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(4):
        q, f = MyDifCode(img, e_values[i], 1)

        if i == 0:
            s = contrasting(f)
            desc = "неквантованный"
        else:
            s = contrasting(q)
            desc = "квантованный"

        axes[i].imshow(s, cmap="gray")
        axes[i].set_title(f"Разностный сигнал ({desc}) при e={e_values[i]}")
    plt.show()


if __name__ == "__main__":
    main()