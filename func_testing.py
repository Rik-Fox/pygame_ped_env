import matplotlib.pyplot as plt
import numpy as np

coeffs = np.ones([1])

n = 5
for i in range(1, n):
    coeffs = np.insert(coeffs, 0, 1 / (i + 1))
    coeffs = np.append(coeffs, i + 1)

x = np.linspace(0.0, 1.0, num=1000)
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

Y = []
y_diff = []

for k, i in enumerate(coeffs):
    Y.append(np.power(x, i))
    axs[0][0].plot(x, Y[k], label=f"x^{i}")

for k in range(len(Y)):
    if k < 5:
        y_diff.append(Y[k] - Y[4])
    else:
        y_diff.append(Y[4] - Y[k])

    axs[1][0].plot(x, y_diff[k], label=f"y=x^{i} - y=x")

for k in range(n):
    axs[0][1].plot(x, np.subtract(Y[4 - k], Y[4 + k]), label=f"x^{k+1} - x^1/{k+1}")
    axs[1][1].plot(
        x,
        np.subtract(y_diff[4 - k], y_diff[4 + k]),
        label=f"x^{coeffs[4-k]} - x^{coeffs[4+k]}",
    )


plt.legend()
plt.show()
