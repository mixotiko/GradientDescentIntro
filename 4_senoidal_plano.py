import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(-6, 6, 500)
y = np.sin(1.5*np.sqrt(x**2))+0.1*x**2

plt.plot(x, y, label="Proyección")
plt.title("Proyección de la superficie sobre plano perpendicular")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc='best')
plt.show()
