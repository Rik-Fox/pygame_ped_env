import matplotlib.pyplot as plt
import numpy as np

coeffs = np.ones([1])

n = 6
for i in range(1, n):
    coeffs = np.insert(coeffs, 0, 1 / (i + 1))
    coeffs = np.append(coeffs, i + 1)

x = np.linspace(0.0, 1.0, num=10000)


# using the variable axs for multiple Axes
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

Y = []
y_diff = []

for k, i in enumerate(coeffs):
    Y.append(np.power(x, i))
    axs[0][0].plot(x, Y[k], label=f"x^{i}")

for k in range(len(Y)):
    if k < 5:
        y_diff.append(np.subtract(Y[k], Y[5]))
    else:
        y_diff.append(np.subtract(Y[k], Y[5]))

    axs[1][0].plot(
        x, y_diff[k], label=f"x^{np.round(coeffs[k], 2)} - x^{np.round(coeffs[5], 2)}"
    )

for k in range(n):
    axs[0][1].plot(
        x,
        np.divide(np.add(Y[n - 1 - k], Y[n - 1 + k]), 2),
        label=f"(x^{coeffs[n - 1 - k]} + x^{coeffs[n - 1 - k]})/2",
    )

    axs[1][1].plot(
        x,
        np.divide(np.add(y_diff[n - 1 - k], y_diff[n - 1 + k]), 2),
        label=f"(x^{np.round(coeffs[n - 1 - k], 2)} + x^{np.round(coeffs[n - 1 + k], 2)})/2",
    )

plt.legend()
plt.show()
