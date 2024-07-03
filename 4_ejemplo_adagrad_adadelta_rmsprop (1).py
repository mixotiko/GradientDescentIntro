import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time


start_x_k = 5
start_y_k = 2  # The algorithm starts at (x,y)=(5,2)
rate = 0.6  # Learning rate
# mom_term = 0.8  # momentum term
precision = 0.000001  # This tells us when to stop the algorithm
start_step_size = 1
max_iters = 5000  # maximum number of iterations
epsilon = 0.00000001  # epsilon 10e-8
p_decay = 0.9  # decay rate
# batch_size = 1  # batch size , sgd method size is 1


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
        # norm
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


def adagrad_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate, epsilon
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    grad_sq_accumulative = 0
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        prev_x_k = cur_x_k  # Store current x value in prev_x
        prev_y_k = cur_y_k
        # calculates squares of gradients accumulative
        grad_sq_accumulative = grad_sq_accumulative + df(prev_x_k, prev_y_k)**2
        update_k = np.diag((np.diag(rate / (epsilon + np.sqrt(grad_sq_accumulative)))) *
                           np.transpose(df(prev_x_k, prev_y_k)))
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - update_k  # update descent
        # norm
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


def rms_prop_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate, epsilon, p_decay
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    expected_grad_sq = 0
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        # calculates expected squares of gradients
        expected_grad_sq = p_decay*expected_grad_sq + \
            (1-p_decay)*df(cur_x_k, cur_y_k)**2
        print(expected_grad_sq)
        # calculates RMS
        rms_gk = np.sqrt(expected_grad_sq)
        # calculate update
        delta_xk = rate/(epsilon+rms_gk)*df(cur_x_k, cur_y_k)
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - delta_xk  # update descent
        # norm
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


def adadelta_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, epsilon, p_decay
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    expected_grad_sq = 0
    expected_grad_diff = 0
    rms_diff = np.array([0, 0])
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        # calculates expected squares of gradients
        expected_grad_sq = p_decay*expected_grad_sq + \
            (1-p_decay)*df(cur_x_k, cur_y_k)**2
        # calculates RMS gradient
        rms_gk = np.sqrt(expected_grad_sq+epsilon)
        # compute update
        delta_xk = (rms_diff+epsilon)/(rms_gk)*df(cur_x_k, cur_y_k)
        print(delta_xk)
        # calculates accumulate update
        expected_grad_diff = p_decay*expected_grad_diff+(1-p_decay)*delta_xk**2

        # calculates RMS update
        rms_diff = np.sqrt(expected_grad_diff+epsilon)
        # apply update
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - delta_xk  # update descent
        # norm
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
x_adagrad, y_adagrad, gr_adagrad = adagrad_gd()
toc = time.perf_counter()
adagrad_runtime = toc-tic

tic = time.perf_counter()
x_rms, y_rms, gr_rms = rms_prop_gd()
toc = time.perf_counter()
rms_runtime = toc-tic

tic = time.perf_counter()
x_adadelta, y_adadelta, gr_adadelta = adadelta_gd()
toc = time.perf_counter()
adadelta_runtime = toc-tic

print("Elapsed time GD: "+str(gd_runtime))
print("Elapsed time Adagrad: "+str(adagrad_runtime))
print("Elapsed time RMS: "+str(rms_runtime))
print("Elapsed time Adadelta: "+str(adadelta_runtime))
x = np.linspace(0.5, 5.5, 100)
y = np.linspace(-1.1, 2.1, 100)
fun_map = fill_map(x, y)
n = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

fig = plt.figure(figsize=(10, 6))
# plt.xticks(np.arange(0, 3.5, 0.5))
ax = fig.add_subplot(xlabel='x', ylabel='y')
im = ax.imshow(fun_map, extent=(
    x[0], x[-1], y[0], y[-1]), origin='lower', cmap=plt.cm.viridis, label="function")
# for i, txt in enumerate(n):
#ax.annotate(txt, (x_adadelta[i], y_adadelta[i]))
#ax.annotate(txt, (x_rms[i]-0.2, y_rms[i]), color='white')

ax.text(0.6, -1, 'Nº iteraciones GD:' + str(len(gr_gd))+'\nNº iteraciones Adagrad:' +
        str(len(gr_adagrad))+'\nNº iteraciones RMS:' +
        str(len(gr_rms))+'\nNº iteraciones Adadelta:' +
        str(len(gr_adadelta)), style='italic', bbox={'facecolor': 'green', 'alpha': 0.7})
# ax.text(0.6, -1.8, 'Nº iteraciones GD:' + str(len(gr_gd))+'\nNº iteraciones Momentum:' +
#        str(len(gr_mom)), style='italic', bbox={'facecolor': 'green', 'alpha': 0.5})

bar = fig.colorbar(im)
bar.set_label("Función")

#ax.plot(x_gd, y_gd, 'o-', color='b', label="GD")
ax.plot(x_adagrad, y_adagrad, 'o-', color='orange', label="Adagrad")
ax.plot(x_rms, y_rms, 'o-', label="RMS")
ax.plot(x_adadelta, y_adadelta, '-', color='red', label="Adadelta")
ax.plot(5, 2, 'd', color='white', label='Punto inicial')
ax.plot(2, 1/24, 'd', color='black', label='Mínimo global')
plt.autoscale(enable=True, axis='x', tight=True)
ax.legend(loc='upper left')
plt.show()

max_iters_showed = min(250, len(gr_gd), len(
    gr_adagrad), len(gr_rms))  # control value of max iterations showed
# max_iters_showed = min(150, len(gr_gd), len(
#    gr_mom))

fig2 = plt.figure(2, figsize=(8, 6))
plt.xlabel("Iteración")
plt.ylabel("Valor de la norma del gradiente")
# plt.plot(range(0, max_iters_showed),
#         gr_gd[0:max_iters_showed], label="GD", markersize=2)
plt.plot(range(0, max_iters_showed),
         gr_adagrad[0:max_iters_showed], label="Adagrad")
plt.plot(range(0, max_iters_showed),
         gr_rms[0:max_iters_showed], label="RMS")
plt.plot(range(0, max_iters_showed),
         gr_adadelta[0:max_iters_showed], label="Adadelta")

plt.title("Caída del gradiente")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc='upper right')
plt.show()
