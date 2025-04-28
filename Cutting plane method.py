import numpy as np
def simplex(c, A, b):
    """
    Симплекс-метод для задачи: max c^T x, s.t. Ax <= b, x >= 0
    Возвращает: x, значение цели или None если задача неразрешима.
    """
    m, n = A.shape

    # Создание симплекс-таблицы
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    basis = list(range(n, n + m))

    while True:
        # Индекс наибольшего отрицательного коэффициента в строке цели
        pivot_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, pivot_col] >= 0:
            break  # Оптимум найден

        # Выбор ведущей строки (по минимальному отношению)
        ratios = []
        for i in range(m):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios)
        if ratios[pivot_row] == np.inf:
            return None, None  # Неразрешимо

        # Поворот таблицы
        pivot = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        basis[pivot_row] = pivot_col

    # Извлечение решения
    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tableau[i, -1]

    return x, tableau[-1, -1]


def cutting_plane_method(c, A, b, max_iter=100):
    """
    Метод отсеканий с симплекс-методом.
    Ищет целочисленное решение (все x[i] должны быть целыми).
    """
    A = A.copy()
    b = b.copy()
    c = np.array(c)

    for it in range(max_iter):
        x, val = simplex(c, A, b)
        if x is None:
            print("Задача неразрешима.")
            return None, None

        if all(np.isclose(xi, round(xi)) for xi in x):
            print(f"Целочисленное решение найдено на итерации {it+1}")
            return x, val

        # Находим первую нецелую переменную и добавляем отсекание
        for i in range(len(x)):
            if not np.isclose(x[i], round(x[i])):
                floor_val = np.floor(x[i])
                frac = x[i] - floor_val
                a = np.zeros_like(x)
                a[i] = 1
                A = np.vstack([A, a])
                b = np.append(b, floor_val)
                print(f"Отсекание добавлено: x[{i}] <= {floor_val}")
                break

    print("Максимальное число итераций достигнуто.")
    return x, val

if __name__ == "__main__":
    # max z = 3x1 + 2x2
    # s.t.
    #    2x1 + x2 <= 18
    #    2x1 + 3x2 <= 42
    #    3x1 + x2 <= 24
    #    x1, x2 >= 0

    c = [3, 2]
    A = np.array([
        [2, 1],
        [2, 3],
        [3, 1]
    ])
    b = np.array([18, 42, 24])

    x_opt, val_opt = cutting_plane_method(c, A, b)
    print("Оптимальное решение:", x_opt)
    print("Значение функции:", val_opt)
