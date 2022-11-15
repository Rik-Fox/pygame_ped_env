import matplotlib.pyplot as plt
import numpy as np

coeffs = np.ones([1])

n = 5
for i in range(1, n):
    coeffs = np.insert(coeffs, 0, 1 / (i + 1))
    coeffs = np.append(coeffs, i + 1)

x = np.linspace(0.0, 1.0, num=1000)
plt.figure(figsize=(12, 9))

for i in coeffs:
    y = np.power(x, i)
    plt.plot(x, y, label=f"x^{i}")

plt.legend()
plt.show()
