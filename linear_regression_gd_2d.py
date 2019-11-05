import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from memory_profiler import profile
from sklearn.datasets.samples_generator import make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import utils
from matplotlib.collections import LineCollection

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.random.seed(0)

x_train, y_train, x_test, y_test = utils.synthetic_linear()
training_size = x_train.shape[0]

w_ols = utils.ordinary_least_squares(x_train, y_train)
y_hat_ols_test = utils.predict(x_test, w_ols)
y_hat_ols_train = utils.predict(x_train, w_ols)

x_norm = (x_train[:, 1] - x_train[:, 1].mean(axis=0))/x_train[:, 1].std(axis=0)
_w_gd_, _b_gd_, _w_trail_path, _b_trail_path = utils.train_gd_linear(x_norm.reshape(training_size, 1), y_train)
y_hat_gd_train = utils.predict_with_bias(x_norm[0:training_size].reshape(training_size, 1), _w_gd_, _b_gd_)

se_ols = np.square(y_hat_ols_train - y_train).sum() / (2 * training_size)
se_gd = np.square(y_hat_ols_train - y_train).sum() / (2 * training_size)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

segments = [[[x_train[:, 1][i], y_train[i]], [x_train[:, 1][i], y_hat_ols_train[i]]] for i in range(training_size)]
lc = LineCollection(segments, zorder=0, color='green')
lc.set_array(np.ones(len(y_train)))
lc.set_linewidths(np.full(40, 0.5))

ax.scatter(x_train[:, 1], y_train, color='blue', label='Data')
ax.scatter(x_train[:, 1], y_hat_gd_train, color='red', label= 'GD')
ax.plot(x_train[:, 1], y_hat_gd_train, color='green', label= 'GD')

ax.set_title('Predictions by OLS')
plt.gca().add_collection(lc)
ax.legend()

ax = fig.add_subplot(1, 3, 2)

delta = 0.10
_w_ = np.arange(-10, 10, delta)
_w_0 = np.arange(-10, 10, delta)
W, W_0 = np.meshgrid(_w_, _w_0)

# Calculate loss surface
loss = []
for w, b in np.nditer([W, W_0]):
    _y_hat = utils.predict_with_bias(x_train[:, 1].reshape(training_size, 1), w.reshape(1), b.reshape(1))
    loss.append(np.square(_y_hat - y_train).sum()  /(2 * training_size))
loss_ = np.array(loss).reshape(W.shape)


ax.contour(W_0, W, loss_)
ax.scatter(_b_gd_, _w_gd_, se_gd, label='GD')
ax.scatter(w_ols[0], w_ols[1], se_ols, label='OLS')
ax.legend()
ax.set_title('Countor plot of Loss surface')

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(W_0, W, loss_, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.75)

ax.scatter(_b_gd_, _w_gd_, se_gd, color='green', s=30, label='GD')
ax.scatter(w_ols[0], w_ols[1], se_ols, color='purple', s=30, label='OLS')
ax.set_title('3-D Loss surface')
ax.view_init(22, 171)
ax.legend()

plt.savefig('loss_surface_2d_ols_gd.eps', format='eps')

plt.show()