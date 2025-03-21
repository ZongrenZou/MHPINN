import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import time
import argparse

import models
import flows


def main(l, idx, flow_name):
    file_name = "./data/data_" + str(int(100*l)) + ".mat"
    data = sio.loadmat(file_name)
    t = data["t"]
    u_ref = data["sols"]
    f = data["f"]
    l = data["l"]

    t_train = t[::8]
    f_train = f[::8, :2000]
    u_ref = u_ref[..., :2000]

    model_name = "meta_" + str(int(100 * l)) + "_" + str(int(idx))
    meta = models.Meta(num_tasks=2000, dim=50, name=model_name)
    meta.restore()

    
    t_test = np.linspace(0, 1, 257).reshape([-1, 1])
    u_pred = meta.call(
        tf.constant(t_test, tf.float32), meta.heads
    ).numpy()

    L2 = np.sqrt(np.sum((u_pred - u_ref[:, 0, :])**2, axis=0) / np.sum(u_ref[:, 0, :]**2, axis=0))

    permutation = list(np.arange(26, 51, 1)) + list(np.arange(0, 26, 1))

    model_name = flow_name + "_" + str(int(100*l)) + "_" + str(int(idx))
    if flow_name == "realnvp":
        nf = flows.RealNVP(
            dim=51,
            num_masked=26,
            permutation=permutation,
            hidden_layers=[100, 100],
            num_bijectors=10,
            activation=tf.nn.relu,
            name=model_name,
            eps=0.1,
        )
    elif flow_name == "maf":
        nf = flows.MAF(
            dim=51,
            permutation=permutation,
            hidden_layers=[100, 100],
            num_bijectors=10,
            activation=tf.nn.relu,
            name=model_name,
            eps=0.,
        )
    elif flow_name == "iaf":
        nf = flows.IAF(
            dim=51,
            permutation=permutation,
            hidden_layers=[100, 100],
            num_bijectors=10,
            activation=tf.nn.relu,
            name=model_name,
            eps=0.,
        )
    nf.restore()

    heads = nf.sample(10000)
    t_test = tf.constant(np.linspace(0, 1, 257).reshape([-1, 1]), tf.float32)
    f_pred = meta.pde(t_test, tf.transpose(heads))

    K = tfp.stats.covariance(tf.transpose(f_pred)).numpy() / 256
    eigs = np.real(np.linalg.eigvals(K)).astype(np.float32)
    eigs = eigs[np.argsort(-eigs)]

    K = tfp.stats.covariance(tf.transpose(f)).numpy() / 256
    eigs2 = np.real(np.linalg.eigvals(K)).astype(np.float32)
    eigs2 = eigs2[np.argsort(-eigs2)]
    return eigs, eigs2

    # plt.plot(eigs[:15], 'o-', label="learned")
    # # plt.plot(eigs3[::-1][:20], 's-', label="training")
    # plt.plot(eigs2[:15], 's-', label="reference")
    # plt.legend()
    # plt.title("Top 15 eigenvalues")

if __name__ == "__main__":
    l = 0.1
    eigs_realnvp = []
    for i in range(1):
        eigs_realnvp += [main(l, i, "realnvp")[0].reshape([-1, 1])]
    eigs_ref = main(l, 0, "realnvp")[1]

    eigs_realnvp = np.concatenate(eigs_realnvp, axis=-1)
    print(eigs_realnvp.shape)
    mu = np.mean(eigs_realnvp, axis=-1)
    std = np.std(eigs_realnvp, axis=-1)

    plt.plot(np.arange(0, 15).astype(np.int32), eigs_ref[:15], "-s", color="black", label="reference")
    plt.errorbar(np.arange(0, 15).astype(np.int32), mu[:15], std[:15], fmt="-o", label="realnvp")
    plt.legend()
    plt.show()


