import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import scipy.io as sio
import time

import models
import flows


data = sio.loadmat("./data/data.mat")
xx = data["xx"]
yy = data["yy"]
x = data["x"]
y = data["y"]
f = data["f"]
u = data["u"]


x_test = x
y_test = y
xx_test, yy_test = np.meshgrid(x_test, y_test)

x_train = x[:, ::2]
y_train = y[:, ::2]
xx_train, yy_train = np.meshgrid(x_train, y_train)
N = 5000
f_train = f[::2, ::2, :N]


x_train = xx_train.reshape([-1, 1])
y_train = yy_train.reshape([-1, 1])
f_train = f_train.reshape([-1, N])


mhpinn = models.MHPINN(num_tasks=N, dim=100, eps=0.0, name="mhpinn_nv")

t0 = time.time()
loss = mhpinn.train(
    x_train, 
    y_train, 
    f_train, 
    niter=50000, 
    ftol=1e-7,
)
t1 = time.time()
print(t1 - t0)


out = mhpinn.call(xx_test.reshape([-1, 1]), yy_test.reshape([-1, 1])).numpy()
out = out.reshape([101, 101, N])
L2 = np.sqrt(np.sum(np.sum((out - u[..., :N])**2, axis=0), axis=0) / np.sum(np.sum(u[..., :N]**2, axis=0), axis=0))
print(np.mean(L2), np.std(L2))
