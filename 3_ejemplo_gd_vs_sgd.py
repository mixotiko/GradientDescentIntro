import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time


start_x_k = 5
start_y_k = 2  # The algorithm starts at (x,y)=(5,2)
rate = 0.08  # Learning rate
precision = 0.000001  # This tells us when to stop the algorithm
start_step_size = 1
max_iters = 1000  # maximum number of iterations
batch_size = 1  # batch size , sgd method size is 1


def my_function(x, y):
    #
    return 0.25*x**2 + 12*y**2-x-y


def fill_map(x, y):
    fun_map = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            fun_map[j, i] = my_function(x[i], y[j])
            # if fun_map[j, i] < 1:
            #    print("Coordenadas que se pasan:",
            #          x[i], ",", y[j], ",", fun_map[i, j])
    return fun_map


def df(x, y): return np.array([0.5*x-1,
                               24*y-1])  # Gradient of our function


def gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    gradients_iter = []
    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        prev_x_k = cur_x_k  # Store current x value in prev_x
        prev_y_k = cur_y_k
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - rate * df(prev_x_k, prev_y_k)  # Grad descent
        cal_gr_x, cal_gr_y = df(cur_x_k, cur_y_k)
        previous_step_size = math.sqrt((cal_gr_x)**2 +
                                       cal_gr_y**2)  # Change in x
        iters = iters+1  # iteration count
        gradients_iter.append(previous_step_size)
        print("Iteration", iters, "\nX value is (", cur_x_k,
              " ,", cur_y_k, " )")  # Print iterations

    print("The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
    x_pred = np.array(x_pred)
    y_pred = np.array(y_pred)

    return x_pred, y_pred, gradients_iter


def sgd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    # 2 variables, one of them is taken as a gradient and the rest of variables takes 0
    stochastic_gradiend = np.array([1, 0])
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        random.shuffle(stochastic_gradiend)  # random choice of subgradient
        prev_x_k = cur_x_k  # Store current x value in prev_x
        prev_y_k = cur_y_k
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - rate*stochastic_gradiend * df(prev_x_k, prev_y_k)  # Grad descent
        cal_gr_x, cal_gr_y = df(cur_x_k, cur_y_k)
        previous_step_size = math.sqrt((cal_gr_x)**2 +
                                       cal_gr_y**2)  # Change in x
        iters = iters+1  # iteration count
        gradients_iter.append(previous_step_size)
        print("Iteration", iters, "\nX value is (", cur_x_k,
              " ,", cur_y_k, " )")  # Print iterations

    print("The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
    x_pred = np.array(x_pred)
    y_pred = np.array(y_pred)

    return x_pred, y_pred, gradients_iter


tic = time.perf_counter()
x_gd, y_gd, gr_gd = gd()
toc = time.perf_counter()
gd_runtime = toc-tic

tic = time.perf_counter()
x_sgd1, y_sgd1, gr_sgd1 = sgd()
toc = time.perf_counter()
sgd1_runtime = toc-tic

tic = time.perf_counter()
x_sgd2, y_sgd2, gr_sgd2 = sgd()
toc = time.perf_counter()
sgd2_runtime = toc-tic

tic = time.perf_counter()
x_sgd3, y_sgd3, gr_sgd3 = sgd()
toc = time.perf_counter()
sgd3_runtime = toc-tic

print("Elapsed time GD: "+str(gd_runtime))
print("Elapsed time SGD-1: "+str(sgd1_runtime))
print("Elapsed time SGD-2: "+str(sgd2_runtime))
print("Elapsed time SGD-3: "+str(sgd3_runtime))
x = np.linspace(1.5, 5.5, 100)
y = np.linspace(-2, 2.1, 100)
fun_map = fill_map(x, y)
n = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

fig = plt.figure(figsize=(10, 6))
# plt.xticks(np.arange(0, 3.5, 0.5))
ax = fig.add_subplot(xlabel='x', ylabel='y')
im = ax.imshow(fun_map, extent=(
    x[0], x[-1], y[0], y[-1]), origin='lower', cmap=plt.cm.viridis, label="function")
# for i, txt in enumerate(n):
#    ax.annotate(txt, (x_gd[i]-0.2, y_gd[i]-0.1))

ax.text(1.6, -1.8, 'Nº iteraciones GD:' + str(len(gr_gd))+'\n Nº iteraciones SGD-1:' + str(len(gr_sgd1))+'\n Nº iteraciones SGD-2:' +
        str(len(gr_sgd2))+'\n Nº iteraciones SGD-3:' + str(len(gr_sgd3)), style='italic', bbox={'facecolor': 'green', 'alpha': 0.5})

bar = fig.colorbar(im)
bar.set_label("Función")
# ax.plot(x, y, 'r', label="functi")
ax.plot(x_gd, y_gd, 'o-', color='b', label="GD")
ax.plot(x_sgd1, y_sgd1, 'o-', color='red', label="SGD-1")
ax.plot(x_sgd2, y_sgd2, 'o-', color='green', label="SGD-2")
ax.plot(x_sgd3, y_sgd3, 'o-', color='orange', label="SGD-3")
ax.plot(5, 2, 'd', color='white', label='Punto inicial')
ax.plot(2, 1/24, 'd', color='black', label='Mínimo global')
plt.autoscale(enable=True, axis='x', tight=True)
ax.legend(loc='upper left')
plt.show()

max_iters_showed = min(100, len(gr_gd), len(
    gr_sgd1), len(gr_sgd2), len(gr_sgd3))  # control value of max iterations showed

fig2 = plt.figure(2, figsize=(8, 6))
plt.xlabel("Iteración")
plt.ylabel("Valor de la norma del gradiente")
plt.plot(range(0, max_iters_showed),
         gr_gd[0:max_iters_showed], label="GD", markersize=2)
plt.plot(range(0, max_iters_showed),
         gr_sgd1[0:max_iters_showed], label="SGD-1")
plt.plot(range(0, max_iters_showed),
         gr_sgd2[0:max_iters_showed], label="SGD-2")
plt.plot(range(0, max_iters_showed),
         gr_sgd3[0:max_iters_showed], label="SGD-3")
plt.title("Caída del gradiente")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc='upper right')
plt.show()
