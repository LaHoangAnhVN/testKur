import math
import random
import numpy as np
import matplotlib.pyplot as plt


a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
plt.subplot(121)
x = np.arange(0, len(a), 1)
plt.plot(x, a)
plt.subplot(122)
plt.plot(x, b)
plt.show()

