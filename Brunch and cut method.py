import numpy as np
def simplex(c, A, b):
    """
    Симплекс-метод: max c^T x, s.t. Ax <= b, x >= 0
    Возвращает: x, значение цели или None если задача неразрешима.
    """
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    basis = list(range(n, n + m))

    while True:
        pivot_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, pivot_col] >= 0:
            break

        ratios = []
        for i in range(m):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios)
        if ratios[pivot_row] == np.inf:
            return None, None

        pivot = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        basis[pivot_row] = pivot_col

    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tableau[i, -1]

    return x, tableau[-1, -1]


def is_integer_vector(x, tol=1e-5):
    return all(abs(xi - round(xi)) <= tol for xi in x)


def branch_and_cut(c, A, b, best_val=float('-inf'), best_x=None, depth=0, max_depth=20):
    """
    Метод ветвлений и отсечений для задачи целочисленного линейного программирования.
    Возвращает: лучшее целочисленное решение x и значение цели.
    """
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    indent = "  " * depth
    x, val = simplex(c, A, b)

    if x is None:
        print(f"{indent}Подзадача неразрешима")
        return best_x, best_val

    if val <= best_val:
        print(f"{indent}Отсечено по ограничению: {val} <= {best_val}")
        return best_x, best_val

    if is_integer_vector(x):
        print(f"{indent}Целое решение: {x}, цель: {val}")
        if val > best_val:
            return x, val
        return best_x, best_val

    if depth >= max_depth:
        print(f"{indent}Превышена глубина рекурсии.")
        return best_x, best_val

    # Находим переменную с наибольшей дробной частью
    frac_parts = [abs(xi - round(xi)) for xi in x]
    i = np.argmax(frac_parts)
    xi = x[i]
    floor_val = np.floor(xi)
    ceil_val = np.ceil(xi)

    print(f"{indent}Ветвление по x[{i}] = {xi:.4f}")

    # Ветвление 1: x[i] <= floor(xi)
    A1 = np.vstack([A, np.eye(len(x))[i]])
    b1 = np.append(b, floor_val)
    best_x, best_val = branch_and_cut(c, A1, b1, best_val, best_x, depth + 1, max_depth)

    # Ветвление 2: x[i] >= ceil(xi) -> -x[i] <= -ceil(xi)
    A2 = np.vstack([A, -np.eye(len(x))[i]])
    b2 = np.append(b, -ceil_val)
    best_x, best_val = branch_and_cut(c, A2, b2, best_val, best_x, depth + 1, max_depth)

    return best_x, best_val
