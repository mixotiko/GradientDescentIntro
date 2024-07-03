import numpy as np
import matplotlib.pyplot as plt
import math


def my_function(x, y):
    #
    return 0.3*x**2 + 12*y**2-x-y


def fill_map(x, y):
    fun_map = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            print(x[i], j[i], my_function(x[i], y[j]))
            fun_map[i, j] = my_function(x[i], y[j])
            if fun_map[i, j] < 1:
                print("Coordenadas que se pasan:",
                      x[i], ",", y[j], ",", fun_map[i, j])
    return fun_map


x = np.linspace(1, 5.5, 20)
y = np.linspace(-2.1, 2.1, 20)

fill_map(x, y)
