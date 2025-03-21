import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
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

    if flow_name == "realnvp":
        # train and load
        print("Training the multi-head NN...")
        t0 = time.time()
        loss = meta.train(t_train, f_train, niter=50000, ftol=1e-10)
        t1 = time.time()
        print("Elapsed: ", t1 - t0)
        meta.restore()
    else:
        # directly load
        print("No training on the multi-head NN")
        t1 = t0 = 0
        meta.restore()
    meta.restore()

    
    t_test = np.linspace(0, 1, 257).reshape([-1, 1])
    u_pred = meta.call(
        tf.constant(t_test, tf.float32), meta.heads
    ).numpy()

    L2 = np.sqrt(np.sum((u_pred - u_ref[:, 0, :])**2, axis=0) / np.sum(u_ref[:, 0, :]**2, axis=0))
    print(np.mean(L2))

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

    heads = meta.heads.numpy().T
    t2 = time.time()
    loss = nf.train_batch(tf.constant(heads, tf.float32), nepoch=500, batch_size=100)
    t3 = time.time()

    print("Elapsed: ", t3 - t2)


    np.savetxt(
        "outputs/" + flow_name + "_time_" + str(int(100 * l)) + "_" + str(int(idx)) + ".txt",
        np.array([t1 - t0, t3 - t2, np.mean(L2)]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=float, default=0.10, help="length scale of GP")
    parser.add_argument(
        "--idx", type=float, default=0, help="index of the current experiment",
    )
    parser.add_argument(
        "--flow_name", type=str, default="realnvp", help="the name of the normalizing flow",
    )

    args = parser.parse_args()

    l = args.l
    idx = args.idx
    flow_name = args.flow_name

    main(l, idx, flow_name)
