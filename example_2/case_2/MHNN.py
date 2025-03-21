import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
import time
import argparse

import models
import flows


def main(KL=4):
    file_name = "./dataset/data_" + str(int(KL)) + ".mat"
    data = sio.loadmat(file_name)
    t = data["t"]
    u_ref = data["sols"]
    f = data["f"]
    k = data["k"]

    t_f_train = t[::8]
    f_train = f[::8, :2000]
    t_u_train = t[::32]
    u_train = u_ref[::32, 0, :2000]
    u_exact = u_ref[:, 0, :2000]
    k_exact = k[:, :2000]

    model_name = "meta_KL_" + str(int(KL))
    meta = models.MHPINN(num_tasks=2000, dim=50, name=model_name)
    t0 = time.time()
    loss = meta.train(t_f_train, f_train, t_u_train, u_train, niter=50000, ftol=1e-10)
    t1 = time.time()
    meta.restore()
    L2 = tf.math.abs(tf.math.exp(meta.log_ks) - k_exact) / k_exact
    print(np.mean(L2))

    permutation = list(np.arange(26, 52, 1)) + list(np.arange(0, 26, 1))

    nf = flows.MAF(
        dim=52,
        permutation=permutation,
        hidden_layers=[100, 100],
        num_bijectors=10,
        activation=tf.nn.relu,
        name="maf_KL_" + str(int(KL)),
    )
    heads = meta.heads.numpy().T
    log_ks = meta.log_ks.numpy().T
    data = np.concatenate([heads, log_ks], axis=-1)
    t2 = time.time()
    loss = nf.train_batch(tf.constant(data, tf.float32), nepoch=500)
    t3 = time.time()
    nf.restore()

    # test
    samples = nf.sample(10000)
    heads = samples[:, 0:-1]
    log_ks = samples[:, -1:]

    t_test = tf.constant(np.linspace(0, 1, 257).reshape([-1, 1]), tf.float32)

    u_pred = meta.call(t_test, tf.transpose(heads))
    for i in range(1000):
        plt.plot(t_test, u_pred[:, i])
    plt.ylim([-1.0, 1])
    plt.show()

    plt.figure()
    for i in range(1000):
        plt.plot(t_test, u_ref[:, 0, i])
    plt.ylim([-1.0, 1])
    plt.show()

    f_pred = meta.pde(t_test, tf.transpose(heads), tf.transpose(log_ks))
    for i in range(1000):
        plt.plot(t_test, f_pred[:, i])
    plt.ylim([-4.0, 4])
    plt.show()

    plt.figure()
    for i in range(1000):
        plt.plot(t_test, f[:, i])
    plt.ylim([-4.0, 4])
    plt.show()

    K = tfp.stats.covariance(tf.transpose(f_pred)).numpy() / 256
    eigs = np.real(np.linalg.eigvals(K)).astype(np.float32)
    eigs = eigs[np.argsort(-eigs)]

    K = tfp.stats.covariance(tf.transpose(f)).numpy() / 256
    eigs2 = np.real(np.linalg.eigvals(K)).astype(np.float32)
    eigs2 = eigs2[np.argsort(-eigs2)]

    plt.plot(eigs[:15], '-o')
    plt.plot(eigs2[:15], '-s')
    plt.show()

    plt.hist(samples[:, -1], density=True, alpha=0.3)
    plt.hist(meta.log_ks.numpy().flatten(), density=True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl", type=int, default=4, help="number of terms being kept")

    args = parser.parse_args()

    main(args.kl)
