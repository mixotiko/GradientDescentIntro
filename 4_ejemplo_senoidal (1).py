import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time


start_x_k = 6
start_y_k = 4.8  # The algorithm starts at (x,y)=(5,2.5)
rate = 0.3  # Learning rate
mom_term = 0.8  # momentum term
precision = 0.000001  # This tells us when to stop the algorithm
start_step_size = 1
max_iters = 10000  # maximum number of iterations
epsilon = 0.00000001  # epsilon 10e-8
p_decay = 0.9  # decay rate
beta_1 = 0.9  # decay rate first order
beta_2 = 0.99  # decay rate second order
# batch_size = 1  # batch size , sgd method size is 1


def my_function(x, y):
    #
    # return 0.25*x**2 + 12*y**2-x-y
    return math.sin(1.5*math.sqrt(x**2+y**2))+0.1*x**2+0.1*y**2


def fill_map(x, y):
    fun_map = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            fun_map[j, i] = my_function(x[i], y[j])
            # if fun_map[j, i] < 1:
            #    print("Coordenadas que se pasan:",
            #          x[i], ",", y[j], ",", fun_map[i, j])
    return fun_map


def df(x, y): return np.array([x*(1.5*math.cos(1.5*math.sqrt(x**2+y**2))/math.sqrt(x**2+y**2)+0.2),
                               y*(1.5*math.cos(1.5*math.sqrt(x**2+y**2))/math.sqrt(x**2+y**2)+0.2)])  # Gradient of our function


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
     #   print("Iteration", iters, "\nX value is (", cur_x_k,
     #         " ,", cur_y_k, " )")  # Print iterations

    print("GD: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
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
      #  print("Iteration", iters, "\nX value is (", cur_x_k,
     #         " ,", cur_y_k, " )")  # Print iterations

    print("SGD: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
    x_pred = np.array(x_pred)
    y_pred = np.array(y_pred)

    return x_pred, y_pred, gradients_iter


def momentum_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    v_k = np.array([0, 0])  # calculates the step depth
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        prev_x_k = cur_x_k  # Store current x value in prev_x
        prev_y_k = cur_y_k
        prev_v_k = v_k
        # calculates momentum
        v_k = mom_term*prev_v_k + rate * df(prev_x_k, prev_y_k)
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - v_k  # Grad descent
        # norm
        cal_gr_x, cal_gr_y = df(cur_x_k, cur_y_k)
        previous_step_size = math.sqrt((cal_gr_x)**2 +
                                       cal_gr_y**2)  # Change in x
        iters = iters+1  # iteration count
        gradients_iter.append(previous_step_size)
   #     print("Iteration", iters, "\nX value is (", cur_x_k,
  #            " ,", cur_y_k, " )")  # Print iterations

    print("Momentum: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
    x_pred = np.array(x_pred)
    y_pred = np.array(y_pred)

    return x_pred, y_pred, gradients_iter


def nag_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    v_k = np.array([0, 0])  # calculates the step depth
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        prev_v_k = v_k
        # Store and calculate predicted step with current x_k value and v_k
        nag_predict_x_k, nag_predict_y_k = np.array(
            [cur_x_k, cur_y_k])-mom_term*prev_v_k

        # calculates nesterov momentum
        v_k = mom_term*prev_v_k + rate * df(nag_predict_x_k, nag_predict_y_k)
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - v_k  # Grad descent
        # norm
        cal_gr_x, cal_gr_y = df(cur_x_k, cur_y_k)
        previous_step_size = math.sqrt((cal_gr_x)**2 +
                                       cal_gr_y**2)  # Change in x
        iters = iters+1  # iteration count
        gradients_iter.append(previous_step_size)
        # if iters < 50:
        #       print("Iteration", iters, "\nX value is (", cur_x_k,
        #              " ,", cur_y_k, " )\nNAG is ", v_k, " with predicted step: (", nag_predict_x_k, ",", nag_predict_y_k, ")")  # Print iterations

    print("NAG: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
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
  #      print("Iteration", iters, "\nX value is (", cur_x_k,
 #             " ,", cur_y_k, " )")  # Print iterations

    print("Adagrad: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
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
        # print(expected_grad_sq)
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
      #  print("Iteration", iters, "\nX value is (", cur_x_k,
     #         " ,", cur_y_k, " )")  # Print iterations

    print("RMS: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
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
        # print(delta_xk)
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
     #   print("Iteration", iters, "\nX value is (", cur_x_k,
    #          " ,", cur_y_k, " )")  # Print iterations

    print("Adadelta: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
    x_pred = np.array(x_pred)
    y_pred = np.array(y_pred)

    return x_pred, y_pred, gradients_iter


def adam_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate, epsilon, beta_1, beta_2
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    m_k = 0
    v_k = 0
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        iters = iters+1  # iteration count
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        # calculates expected squares of gradients
        g_k = df(cur_x_k, cur_y_k)
        m_k = beta_1*m_k + \
            (1-beta_1)*g_k
        v_k = beta_2*v_k + \
            (1-beta_2)*g_k**2

        # calculate corrections
        mhat_k = m_k/(1-beta_1**iters)
        vhat_k = v_k/(1-beta_2**iters)
        # calculates RMS
        rms_vk = np.sqrt(vhat_k)
        # calculate update
        delta_xk = rate/(epsilon+rms_vk)*mhat_k
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - delta_xk  # update descent
        # norm
        cal_gr_x, cal_gr_y = df(cur_x_k, cur_y_k)
        previous_step_size = math.sqrt((cal_gr_x)**2 +
                                       cal_gr_y**2)  # Change in x

        gradients_iter.append(previous_step_size)
  #      print("Iteration", iters, "\nX value is (", cur_x_k,
 #             " ,", cur_y_k, " )")  # Print iterations

    print("Adam: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
    x_pred = np.array(x_pred)
    y_pred = np.array(y_pred)

    return x_pred, y_pred, gradients_iter


def nadam_gd():
    global start_step_size, start_x_k, start_y_k, max_iters, precision, rate, epsilon, beta_1, beta_2
    x_pred = []
    y_pred = []
    previous_step_size = start_step_size
    iters = 0  # iteration counter
    cur_x_k = start_x_k
    cur_y_k = start_y_k
    m_k = 0
    v_k = 0
    mhat_k = 0
    gradients_iter = []

    while previous_step_size > precision and iters < max_iters:
        iters = iters+1  # iteration count
        x_pred.append(cur_x_k)
        y_pred.append(cur_y_k)
        # calculates expected squares of gradients
        g_k = df(cur_x_k, cur_y_k)
        m_k = beta_1*m_k + \
            (1-beta_1)*g_k
        v_k = beta_2*v_k + \
            (1-beta_2)*g_k**2

        # calculate corrections
        previous_mhat_k = mhat_k
        mhat_k = m_k/(1-beta_1**iters)
        vhat_k = v_k/(1-beta_2**iters)
        # calculates RMS
        rms_vk = np.sqrt(vhat_k)
        # calculate update
        delta_xk = rate/(epsilon+rms_vk)*(previous_mhat_k *
                                          beta_1+(1-beta_1)*g_k/(1-beta_1**iters))
        cur_x_k, cur_y_k = np.array(
            [cur_x_k, cur_y_k]) - delta_xk  # update descent
        # norm
        cal_gr_x, cal_gr_y = df(cur_x_k, cur_y_k)
        previous_step_size = math.sqrt((cal_gr_x)**2 +
                                       cal_gr_y**2)  # Change in x

        gradients_iter.append(previous_step_size)
        # print("Iteration", iters, "\nX value is (", cur_x_k,
        #      " ,", cur_y_k, " )")  # Print iterations

    print("Nadam: The local minimum occurs at (", cur_x_k, " ,", cur_y_k, " )")
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
x_mom, y_mom, gr_mom = momentum_gd()
toc = time.perf_counter()
mom_runtime = toc-tic


tic = time.perf_counter()
x_nag, y_nag, gr_nag = nag_gd()
toc = time.perf_counter()
nag_runtime = toc-tic

tic = time.perf_counter()
x_adam, y_adam, gr_adam = adam_gd()
toc = time.perf_counter()
adam_runtime = toc-tic

tic = time.perf_counter()
x_nadam, y_nadam, gr_nadam = nadam_gd()
toc = time.perf_counter()
nadam_runtime = toc-tic

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
print("Elapsed time SGD: "+str(sgd1_runtime))
print("Elapsed time Momentum: "+str(mom_runtime))
print("Elapsed time NAG: "+str(nag_runtime))
print("Elapsed time Adam: "+str(adam_runtime))
print("Elapsed time Nadam: "+str(nadam_runtime))
print("Elapsed time Adagrad: "+str(adagrad_runtime))
print("Elapsed time RMS: "+str(rms_runtime))
print("Elapsed time Adadelta: "+str(adadelta_runtime))

"""

"""

x = np.linspace(-5, 7, 500)
y = np.linspace(-4, 5, 500)
fun_map = fill_map(x, y)
n = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

fig = plt.figure(figsize=(14, 8))
# plt.xticks(np.arange(0, 3.5, 0.5))
ax = fig.add_subplot(xlabel='x', ylabel='y')
im = ax.imshow(fun_map, extent=(
    x[0], x[-1], y[0], y[-1]), origin='lower', cmap=plt.cm.viridis, label="function")
# for i, txt in enumerate(n):
#    ax.annotate(txt, (x_adam[i], y_adam[i]))
#    ax.annotate(txt, (x_rms[i]-0.2, y_rms[i]), color='white')

ax.text(-4.9, -3.9, 'Nº iteraciones GD:' + str(len(gr_gd))+'\nNº iteraciones SGD:' + str(len(gr_sgd1))+'\nNº iteraciones Adam: ' +
        str(len(gr_adam))+'\nNº iteraciones Nadam: '+str(len(gr_nadam)) +
        '\nNº iteraciones Adagrad:' +
        str(len(gr_adagrad))+'\nNº iteraciones RMS:' +
        str(len(gr_rms))+'\nNº iteraciones Adadelta:' +
        str(len(gr_adadelta)) +
        '\nNº iteraciones Momentum:' + str(len(gr_mom))
        + '\nNº iteraciones NAG:' + str(len(gr_nag)),
        style='italic', bbox={'facecolor': 'green', 'alpha': 0.7})
# ax.text(0.6, -1.8, 'Nº iteraciones GD:' + str(len(gr_gd))+'\nNº iteraciones Momentum:' +
#        str(len(gr_mom)), style='italic', bbox={'facecolor': 'green', 'alpha': 0.5})

bar = fig.colorbar(im)
bar.set_label("Función")

ax.plot(x_gd, y_gd, 'o-', color='black', label="GD")
ax.plot(x_sgd1, y_sgd1, 'o-', label="SGD")
ax.plot(x_mom, y_mom, 'o-', label="Momentum")
ax.plot(x_nag, y_nag, 'o-', label="NAG")
ax.plot(x_adam, y_adam, 'o-', label="Adam")
ax.plot(x_nadam, y_nadam, 'o-', label="Nadam")
ax.plot(x_adagrad, y_adagrad, 'o-', label="Adagrad")
ax.plot(x_rms, y_rms, 'o-', label="RMS")
ax.plot(x_adadelta, y_adadelta, 'o-', label="Adadelta")

"""
ax.plot(-1/math.sqrt(2), 1/math.sqrt(2), 'd',
        color='black', label='Mínimo global')
ax.plot(1/math.sqrt(2), -1/math.sqrt(2), 'd',
        color='black', label='Mínimo global')
ax.plot(0, 0, 'd', color='yellow', label='Punto silla')
"""
ax.plot(start_x_k, start_y_k, 'd', color='white', label='Punto inicial')
circle1 = plt.Circle((0, 0), 2.87894, facecolor="none",
                     edgecolor='black', label="Minimo global")
ax.add_patch(circle1)
plt.autoscale(enable=True, axis='x', tight=True)
ax.legend(loc='upper left')
plt.show()
"""
max_iters_showed = min(250, len(gr_gd), len(
    gr_adagrad), len(gr_rms), len(gr_gd), len(
    gr_adam), len(gr_nadam))  # control value of max iterations showed
# max_iters_showed = min(150, len(gr_gd), len(
#    gr_mom))

fig2 = plt.figure(2, figsize=(8, 6))
plt.xlabel("Iteración")
plt.ylabel("Valor de la norma del gradiente")
plt.plot(range(0, max_iters_showed),
         gr_gd[0:max_iters_showed], label="GD", markersize=2)
plt.plot(range(0, max_iters_showed),
         gr_adam[0:max_iters_showed], label="Adam")
plt.plot(range(0, max_iters_showed),
         gr_nadam[0:max_iters_showed], label="Nadam")
plt.plot(range(0, max_iters_showed),
         gr_adagrad[0:max_iters_showed], label="Adagrad")
plt.plot(range(0, max_iters_showed),
         gr_rms[0:max_iters_showed], label="RMS")
plt.plot(range(0, max_iters_showed),
         gr_adadelta[0:max_iters_showed], label="Adadelta")
plt.plot(range(0, max_iters_showed),
         gr_mom[0:max_iters_showed], label="Momentum")
plt.plot(range(0, max_iters_showed),
         gr_nag[0:max_iters_showed], label="NAG")

plt.title("Caída del gradiente")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc='upper right')
plt.show()
"""
