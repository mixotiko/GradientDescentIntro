import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# unknown function
# we solve regression linear for this case
# in our case we find a and b of the equation y_p=a*x+b
# and we find to minimize sse = sum (y_i - y_p(x_i)) = sum (y_i -a*x_i-b)^2
# where (x_i, y_i) \in source subset


def my_function(x, y, x_values, y_values):
    # sse = sum of squared errors
    sse = 0
    # another option sse = (np.sum(y_values -x*x_values)-y)**2
    for i in range(y_values.size):
        # let (x_value, y_value) a value of point cloud we calculate the sse
        # from linear regression
        sse += (y_values[i]-x*x_values[i]-y)**2
    return math.sqrt(sse)


def fill_map(x, y, x_values, y_values):
    fun_map = np.empty((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            fun_map[j, i] = my_function(x[i], y[j], x_values, y_values)
            # if fun_map[j, i] < 1:
            #    print("Coordenadas que se pasan:",
            #          x[i], ",", y[j], ",", fun_map[i, j])
    return fun_map

# gradient above a and b, we need parameters x_i,y_i,y_p


def df(x, y, y_pred):
    # we calcule the gradient of sse
    # error is vector y minus vector y predicted
    # y_pred = np.squeeze(y_pred)
    # y = np.squeeze(y)
    rest = y - y_pred
    # error = np.reshape(rest, (-1, 1))
    # Gradient of our function
    return np.array([np.sum(-np.dot(rest, x)), np.sum(-rest)])


def minMax(set):
    minimun = min(set)
    maximun = max(set)
    set_normalized = []
    for i in range(set.size-1):
        set_normalized.append((set[i]-minimun)/(maximun-minimun))
    return set_normalized


data = pd.read_excel(
    "C:/Users/tony_/Desktop/universidad/5º PCEO/TFGmat/real_estate_madrid_modified.xlsx")

print(data.head(10))
data.info()
data.to_numpy()
x_values = data.pop('sq_mt')
y_values = data.pop('buy_price')
# print(x_values)
# print(y_values)
x_min = min(x_values)
x_max = max(x_values)
x_values = minMax(x_values)
y_min = min(y_values)
y_max = max(y_values)
y_values = minMax(y_values)
# print("****************NORMALIZED**********************")
# print(x_values)
# print(y_values)
x_values = np.array(x_values)
y_values = np.array(y_values)

init_a = 0
init_b = 0
cur_a_k = init_a
cur_b_k = init_b
# The algorithm starts at (a,b)=(2000,7000) es decir we are supposing we
# start with linear regression y= 2000*x+7000
rate = 0.00006  # Learning rate
precision = 0.000001  # This tells us when to stop the algorithm
previous_step_size = 0.2
max_iters = 1000  # maximum number of iterations
iters = 0  # iteration counter
a_pred = []
b_pred = []
gradients_iter = []

while previous_step_size > precision and iters < max_iters:
    a_pred.append(cur_a_k)
    b_pred.append(cur_b_k)
    prev_a_k = cur_a_k  # Store current x value in prev_x
    prev_b_k = cur_b_k
    y_pred = prev_a_k*x_values+prev_b_k
    cur_a_k, cur_b_k = np.array(
        [cur_a_k, cur_b_k]) - rate * df(x_values, y_values, y_pred)  # Grad descent
    # previous_step_size = math.sqrt((cur_a_k - prev_a_k)**2 +
    #                               (cur_b_k - prev_b_k)**2)  # Change in x
    cal_gr_x, cal_gr_y = df(cur_a_k, cur_b_k)
    previous_step_size = math.sqrt((cal_gr_x)**2 +
                                   cal_gr_y**2)  # Change in x
    gradients_iter.append(previous_step_size)
    iters = iters+1  # iteration count
    print("Iteration", iters, "\nX value is (", cur_a_k,
          " ,", cur_b_k, " )")  # Print iterations


print("Information:\nInitial linear regression: Y =", str(init_a), "X+", str(init_b), "\nLearning rate: ", str(rate), "\nMinMax(X) Min ", x_min, ", Max ",
      x_max, "\nMinMax(Y) Min", y_min, ", Max ", y_max)

x_line = np.linspace(0, 1, 100)
y_line_0 = a_pred[0]*x_line + b_pred[0]
y_line_1 = a_pred[1]*x_line + b_pred[1]
y_line_2 = a_pred[2]*x_line + b_pred[2]
y_line_eighth = a_pred[int(len(a_pred)/8)]*x_line+b_pred[int(len(a_pred)/8)]
y_line_quarter = a_pred[int(len(a_pred)/4)]*x_line+b_pred[int(len(a_pred)/4)]
y_line_end = cur_a_k*x_line+cur_b_k
plot1 = plt.figure(1, figsize=(8, 6))
plt.xlabel("Tamaño vivienda (en m^2)")
plt.ylabel("Precio vivienda (en €)")
plt.plot(x_values, y_values, 'co', markersize=2)
plt.plot(x_line, y_line_0, label="Iteración 0")
plt.plot(x_line, y_line_1, label="Iteración 1")
plt.plot(x_line, y_line_2, label="Iteración 2")
plt.plot(x_line, y_line_eighth, label="Iteración " + str(int(len(a_pred)/16)))
plt.plot(x_line, y_line_quarter, label="Iteración " +
         str(int(len(a_pred)/4)), color='b')
plt.plot(x_line, y_line_end, label="Iteración " + str(len(a_pred)), color='m')
plt.legend(loc='upper left')
plt.show()

plot2 = plt.figure(2, figsize=(8, 6))
plt.xlabel("Iteración")
plt.ylabel("Valor de la norma del gradiente")
plt.plot(range(0, iters), gradients_iter, markersize=2)
plt.title("Caída del gradiente")
plt.show()
