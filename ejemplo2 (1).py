import numpy as np
import matplotlib.pyplot as plt
import math


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


cur_x_k = 5
cur_y_k = 2  # The algorithm starts at (x,y)=(5,2)
rate = 0.08  # Learning rate
precision = 0.000001  # This tells us when to stop the algorithm
previous_step_size = 1
max_iters = 1000  # maximum number of iterations
iters = 0  # iteration counter


x_pred = []
y_pred = []

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
    print("Iteration", iters, "\nX value is (", cur_x_k,
          " ,", cur_y_k, " )")  # Print iterations

print("The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
x_pred = np.array(x_pred)
y_pred = np.array(y_pred)
x = np.linspace(1.5, 5.5, 100)
y = np.linspace(-2, 2.1, 100)
fun_map = fill_map(x, y)
n = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']


fig = plt.figure(figsize=(10, 6))
#plt.xticks(np.arange(0, 3.5, 0.5))
ax = fig.add_subplot(xlabel='x', ylabel='y')
im = ax.imshow(fun_map, extent=(
    x[0], x[-1], y[0], y[-1]), origin='lower', cmap=plt.cm.viridis, label="function")
for i, txt in enumerate(n):
    ax.annotate(txt, (x_pred[i]-0.2, y_pred[i]-0.1))

# ax.text(-1, 55, 'Learning rate: ' + str(rate)+'\n Nº iteraciones:' + str(iters), style='italic', bbox={
#        'facecolor': 'green', 'alpha': 0.5})

bar = fig.colorbar(im)
bar.set_label("Función")
#ax.plot(x, y, 'r', label="functi")
ax.plot(x_pred, y_pred, 'o-', color='peru', label="Iteraciones")
ax.plot(2, 1/24, 'd', label='Mínimo global')
plt.autoscale(enable=True, axis='x', tight=True)
ax.legend(loc='upper left')
plt.show()
