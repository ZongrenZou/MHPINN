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
mhpinn.restore()
out = mhpinn.call(
    tf.constant(xx_test.reshape([-1, 1]), tf.float32), 
    tf.constant(yy_test.reshape([-1, 1]), tf.float32),
    mhpinn.heads,
).numpy()
out = out.reshape([101, 101, N])
L2 = np.sqrt(np.sum(np.sum((out - u[..., :N])**2, axis=0), axis=0) / np.sum(np.sum(u[..., :N]**2, axis=0), axis=0))
print(np.mean(L2), np.std(L2))


# xi = meta_model.xi.numpy()
permutation = list(np.arange(51, 101, 1)) + list(np.arange(0, 51, 1))
# permutation = list(np.arange(26, 51, 1)) + list(np.arange(0, 26, 1))

nf = flows.MAF(
    dim=101, 
    permutation=permutation,
    hidden_layers=[100, 100],
    num_bijectors=10,
    activation=tf.nn.relu,
)

data = mhpinn.heads.numpy()
t2 = time.time()
loss = nf.train_batch(tf.constant(data.T, tf.float32), nepoch=500)
t3 = time.time()


# print(t3 - t2)
# print("Elapsed over phase 2: ", t2 - t1)
# print("Total training time: ", t2 - t0)
nf.restore()
sample_fn = tf.function(nf.sample)
xi_samples = sample_fn(N)
x_test = tf.constant(x_train, tf.float32)
y_test = tf.constant(y_train, tf.float32)


basis = mhpinn.shared_nn(tf.concat([x_test, y_test], axis=-1))
xi = tf.transpose(xi_samples)
u_pred = x_test * (1-x_test) * y_test * (1-y_test) * (tf.matmul(basis, xi[:100]) + xi[100:])
u_pred = tf.reshape(u_pred, [51, 51, N])
u_pred = tf.transpose(u_pred, [2, 0, 1])


for i in range(1000):
    plt.plot(u_pred[i, :, 1])
# plt.ylim([-0.003, 0.003])
plt.show()


u_ref = tf.transpose(u, [2, 0, 1])
for i in range(1000):
    plt.plot(u_ref[i, :, 2*1])
# plt.ylim([-0.003, 0.003])
plt.show()
