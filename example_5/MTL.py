import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time

import models


# load the data. It may take a while
data = sio.loadmat("./data/data_120_10000.mat")


lamb = data["lamb"]
sols = data["sols"]
fs = data["fs"]
xx = data["xx"]
yy = data["yy"]
xi = data["xi"]

N = 10000
u_ref = sols[:N, ...]
f_ref = fs[:N, ...]
x_train = xx[::2, ::2].reshape([-1, 1])
y_train = yy[::2, ::2].reshape([-1, 1])
f_train = f_ref[:, ::2, ::2].reshape([N, -1])
f_train = f_train.T


mhpinn = models.MHPINN(
    num_tasks=N, dim=100, name="mhpinn4",
)


################################################
################### First ######################
################################################
t0 = time.time()
loss = mhpinn.train(
    x_train, 
    y_train, 
    f_train, 
    niter=150000, 
    ftol=1e-6,
)
t1 = time.time()
print(t1 - t0)
xx = data["xx"].reshape([-1, 1])
yy = data["yy"].reshape([-1, 1])
xx_test = tf.constant(xx, tf.float32)
yy_test = tf.constant(yy, tf.float32)
u_pred = mhpinn.call(xx_test, yy_test, mhpinn.heads).numpy()
u_pred = u_pred.reshape([101, 101, N])
u_pred = np.transpose(u_pred, [2, 0, 1])
L2 = np.sqrt(np.sum(np.sum((u_ref - u_pred)**2, axis=-1), axis=-1) / np.sum(np.sum(u_ref**2, axis=-1), axis=-1))
print(np.mean(L2), np.std(L2))


################################################
################## Second ######################
################################################
mhpinn.opt.learning_rate = 5e-4
t0 = time.time()
loss = mhpinn.train(
    x_train, 
    y_train, 
    f_train, 
    niter=50000, 
    ftol=1e-6,
)
t1 = time.time()
print(t1 - t0)
xx = data["xx"].reshape([-1, 1])
yy = data["yy"].reshape([-1, 1])
xx_test = tf.constant(xx, tf.float32)
yy_test = tf.constant(yy, tf.float32)
u_pred = mhpinn.call(xx_test, yy_test, mhpinn.heads).numpy()
u_pred = u_pred.reshape([101, 101, N])
u_pred = np.transpose(u_pred, [2, 0, 1])
L2 = np.sqrt(np.sum(np.sum((u_ref - u_pred)**2, axis=-1), axis=-1) / np.sum(np.sum(u_ref**2, axis=-1), axis=-1))
print(np.mean(L2), np.std(L2))


################################################
################### Third ######################
################################################
mhpinn.opt.learning_rate = 1e-4
t0 = time.time()
loss = mhpinn.train(
    x_train, 
    y_train, 
    f_train, 
    niter=50000, 
    ftol=1e-6,
)
t1 = time.time()
print(t1 - t0)
xx = data["xx"].reshape([-1, 1])
yy = data["yy"].reshape([-1, 1])
xx_test = tf.constant(xx, tf.float32)
yy_test = tf.constant(yy, tf.float32)
u_pred = mhpinn.call(xx_test, yy_test, mhpinn.heads).numpy()
u_pred = u_pred.reshape([101, 101, N])
u_pred = np.transpose(u_pred, [2, 0, 1])
L2 = np.sqrt(np.sum(np.sum((u_ref - u_pred)**2, axis=-1), axis=-1) / np.sum(np.sum(u_ref**2, axis=-1), axis=-1))
print(np.mean(L2), np.std(L2))
