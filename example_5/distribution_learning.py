# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time

import models
import flows


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
mhpinn.restore()


################################################
############## Normalizing flows ###############
################################################
permutation = list(np.arange(51, 101, 1)) + list(np.arange(0, 51, 1))
nf = flows.MAF(
    dim=101,
    permutation=permutation,
    hidden_layers=[100, 100],
    num_bijectors=10,
    activation=tf.nn.relu
)
data = mhpinn.heads.numpy()


################################################
################### Train ######################
################################################
loss = nf.train_batch(tf.constant(data.T, tf.float32), nepoch=1000)


# ################################################
# #################### Test ######################
# ################################################
# nf.restore()
# samples = nf.sample(10000)
# xx_test = tf.constant(xx.reshape([-1, 1]), tf.float32)
# yy_test = tf.constant(yy.reshape([-1, 1]), tf.float32)


# u_samples = mhpinn.call(
#     xx_test, yy_test, tf.transpose(samples),
# )
# f_samples = mhpinn.pde(
#     xx_test, yy_test, tf.transpose(samples),
# )
