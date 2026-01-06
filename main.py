import numpy as np
import matplotlib.pyplot as plt

# x_i = number of items ; y_i = their price
x = np.array([1, 3, 5, 6, 10, 100], dtype=float)
y = np.array([3, 10, 20, 40, 100, 1000], dtype=float)

# Training
m = x.shape[0]
w, b = 0.0, 0.0
alpha = 1e-4
iters = 1000000

for _ in range(iters):
    f = w * x + b
    grad_w = (1/m) * np.sum((f - y) * x)
    grad_b = (1/m) * np.sum(f - y)
    w -= alpha * grad_w
    b -= alpha * grad_b

x_i = 15
f_i = w * x_i + b

x = np.append(x, x_i)
y = np.append(y, f_i)

# Run the graph
plt.plot(x, y, 'o', label="data")
xx = np.linspace(x.min(), x.max(), 200)
plt.plot(xx, w * xx + b, label="fit")
plt.plot([x_i], [f_i], 'o', label="prediction")

plt.xlabel("Number of Items")
plt.ylabel("Price of Items")
plt.title("Practice Graph")
plt.legend()

plt.show()
