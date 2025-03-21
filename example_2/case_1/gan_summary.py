import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time

import data as Data
import net


z_dim = 50
x_num = 65
D_dim = x_num
layers_gen_z = [50, 128, 128, 128, 50]
layers_gen_x = [1, 50, 50, 50, 50]
layers_dis = [65, 128, 128, 128, 1]
batch_size = 100




def main():
    data = Data.DataSet(N=2000, batch_size=batch_size)
    sess = tf.Session()
    t_train, f_train = data.minibatch()

    t_pos = tf.constant(t_train, dtype=tf.float32) #[x_num, x_dim]
    z = tf.tile(tf.random_normal(shape=[batch_size, 1, z_dim], dtype=tf.float32), [1, 65, 1])
    t = tf.tile(t_pos[None, :, :], [batch_size, 1, 1])
    
    f_real = tf.placeholder(shape=[None, D_dim], dtype=tf.float32)
    model = net.DNN()
    W_g_z, b_g_z = model.hyper_initial(layers_gen_z)
    W_g_x, b_g_x = model.hyper_initial(layers_gen_x)

    u_fake_z = model.fnn(z, W_g_z, b_g_z, act=tf.tanh)
    u_fake_z = model.fnn(z, W_g_z, b_g_z, act=tf.tanh) 
    u_fake_x = model.fnn(t, W_g_x, b_g_x, act=tf.tanh)
    u_fake = u_fake_x*u_fake_z
    u_fake = tf.reduce_sum(u_fake, axis=-1, keepdims=True)
    u_fake = t**2 * u_fake
    f_fake = model.pdenn(t, u_fake)[:, :, 0]

    W_d, b_d = model.hyper_initial(layers_dis)
    
    dis_fake = model.fnn(f_fake, W_d, b_d, act=tf.nn.leaky_relu)
    dis_real = model.fnn(f_real, W_d, b_d, act=tf.nn.leaky_relu)

    var_list_gen = [W_g_z, b_g_z, W_g_x, b_g_x]
    var_list_dis = [W_d, b_d]
    
    saver_g = tf.train.Saver([weight for weight in W_g_z+b_g_z+W_g_x+b_g_x], max_to_keep=200)
    saver_d = tf.train.Saver([weight for weight in W_d+b_d])

    sess.run(tf.global_variables_initializer())

    K = tfp.stats.covariance(tf.transpose(data.f)) / 256
    print(K.shape)
    K = sess.run(K)
    eigs3 = np.linalg.eigvals(K).astype(np.float32)
    eigs3 = eigs3[np.argsort(-eigs3)]

    # load different GANs to compute the statistics of the eigen-values
    t_pos = tf.constant(data.t, dtype=tf.float32) #[x_num, x_dim]
    z = tf.tile(tf.random_normal(shape=[batch_size, 1, z_dim], dtype=tf.float32), [1, 257, 1])
    t = tf.tile(t_pos[None, :, :], [batch_size, 1, 1])

    u_fake_z = model.fnn(z, W_g_z, b_g_z, act=tf.tanh)
    u_fake_z = model.fnn(z, W_g_z, b_g_z, act=tf.tanh) 
    u_fake_x = model.fnn(t, W_g_x, b_g_x, act=tf.tanh)
    u_fake = u_fake_x*u_fake_z
    u_fake = tf.reduce_sum(u_fake, axis=-1, keepdims=True)
    u_fake = t**2 * u_fake
    f_fake = model.pdenn(t, u_fake)[:, :, 0]

    eigs = []
    for idx in range(10):
        saver_g.restore(sess, "./results_GAN/GAN_10/checkpoints/prior_"+str(int(idx)))
        out_f = []
        for i in range(20):
            out_f += [sess.run(f_fake)]
        f_pred = np.concatenate(out_f, axis=0)
        K = tfp.stats.covariance(tf.transpose(f_pred.T)) / 256
        K = sess.run(K)
        _eigs = np.real(np.linalg.eigvals(K)).astype(np.float32)
        eigs += [_eigs[np.argsort(-_eigs)]]
    eigs = np.stack(eigs, axis=-1)
    mu = np.mean(eigs, axis=-1)
    std = np.std(eigs, axis=-1)
    
    idx = np.arange(0, 15).astype(np.int32)
    plt.plot(idx, eigs3[:15], "s-", label="reference")
    plt.errorbar(idx, mu[:15], std[:15], label="learned")
    # plt.plot(idx, mu[:15], "o-", label="learned")
    plt.legend()
    plt.show()
    
    return eigs, eigs3
    
#     plt.plot(eigs[:15], 'o-', label="learned")
#     # plt.plot(eigs2[:20], 's-', label="training")
#     plt.plot(eigs3[:15], '^-', label="referece")
#     plt.legend()
#     plt.title("Top 20 eigenvalues")


if __name__ == "__main__":
    main()
    
    
