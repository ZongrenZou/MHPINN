import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import models
import flows


# parser = argparse.ArgumentParser()
# parser.add_argument("--l", type=float, default=0.10, help="length scale of GP")
# parser.add_argument(
#     "--id", type=float, default=0, help="index of the current experiment"
# )

# args = parser.parse_args()

# l = args.l
# idx = args.id


def main(l, idx):
    ###################### load data ######################
    file_name = "./data/data_" + str(int(100 * l)) + ".mat"
    data = sio.loadmat(file_name)
    u_ref = data["sols"]
    f = data["f"]
    t = data["t"]

    t_train = t[::4]
    f_train = f[::4, :2000]
    u_ref = u_ref[..., :2000]

    ###################### build and train multi-head NN ######################
    model_name = "meta_" + str(int(100 * l)) + "_" + str(int(idx))
    meta = models.Meta(num_tasks=2000, dim=50, name=model_name)
    t0 = time.time()
    loss = meta.train(t_train, f_train, niter=30000, ftol=1e-10)
    t1 = time.time()
    ###################### restore trained model to the best ######################
    meta.restore()
    print("Elapsed: ", t1 - t0)
    training_iterations = int(len(loss))
    u_pred = meta.call(tf.constant(t, tf.float32), meta.heads).numpy()
    L2 = np.sqrt(
        np.sum((u_pred - u_ref[:, 0, :]) ** 2, axis=0)
        / np.sum(u_ref[:, 0, :] ** 2, axis=0)
    )
    print("Mean L2 relative error: ", np.mean(L2))

    ###################### build a normalizing flow ######################
    model_name = "flow_" + str(int(100 * l)) + "_" + str(int(idx))
    permutation = list(np.arange(26, 51, 1)) + list(np.arange(0, 26, 1))

    nf = flows.MAF(
        dim=51,
        permutation=permutation,
        hidden_layers=[128, 128],
        num_bijectors=10,
        activation=tf.nn.relu,
        name=model_name,
    )
    heads = meta.heads.numpy().T

    ###################### train the normalizing flow ######################
    t2 = time.time()
    loss = nf.train_batch(tf.constant(heads, tf.float32), nepoch=1000)
    t3 = time.time()

    ###################### restore trained model to the best ######################
    nf.restore()

    print("Elapsed: ", t3 - t2)

    np.savetxt(
        "outputs/time_" + str(int(100 * l)) + "_" + str(int(idx)) + ".txt",
        np.array([t1 - t0, t3 - t2, training_iterations, np.mean(L2)]),
    )


if __name__ == "__main__":
    for idx in range(10):
        main(l=0.1, idx=idx)

    for idx in range(10):
        main(l=0.25, idx=idx)
    
    for idx in range(10):
        main(l=0.5, idx=idx)
