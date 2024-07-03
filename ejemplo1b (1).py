import numpy as np
import matplotlib.pyplot as plt

# function y= x^4-2x^3-5x^2+7x+2

cur_x = 3.3  # The algorithm starts at x=3
rate = 0.05  # Learning rate
precision = 0.000001  # This tells us when to stop the algorithm
previous_step_size = 1
max_iters = 1000  # maximum number of iterations
iters = 0  # iteration counter
def df(x): return 4*x**3-6*x**2-10*x+7  # Gradient of our function


x_pred = []

while previous_step_size > precision and iters < max_iters:
    x_pred.append(cur_x)
    prev_x = cur_x  # Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x)  # Grad descent
    previous_step_size = abs(cur_x - prev_x)  # Change in x
    iters = iters+1  # iteration count
    print("Iteration", iters, "\nX value is", cur_x)  # Print iterations

print("The local minimum occurs at", cur_x)
x_pred = np.array(x_pred)
x = np.linspace(-3, 4, 200)
y = x**4-2*x**3-5*x**2+7*x+2
y_pred = x_pred**4-2*x_pred**3-5*x_pred**2+7*x_pred+2
n = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']


fig = plt.figure()
ax = fig.add_subplot(111)
for i, txt in enumerate(n):
    ax.annotate(txt, (x_pred[i], y_pred[i]+3.5))

ax.text(-1, 55, 'Learning rate: ' + str(rate)+'\n Nº iteraciones:' + str(iters), style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5})

ax.plot(x, y, 'r', label="x^4-2x^3-5x^2+7x+2")
ax.plot(x_pred, y_pred, 'o-', label="Iteraciones",)

ax.legend()
plt.show()
