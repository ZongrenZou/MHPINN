import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
import time

import data
import net

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0)


z_dim = 50
x_num = 65
D_dim = x_num
layers_gen_z = [50, 128, 128, 128, 50]
layers_gen_x = [1, 50, 50, 50, 50]
layers_dis = [65, 128, 128, 128, 1]
batch_size = 100


def main(idx=0):
    dataset = data.DataSet(N=2000, batch_size=batch_size)
    sess = tf.Session()
    t_train, f_train = dataset.minibatch()

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
    
    gen_loss = tf.reduce_mean(dis_real - dis_fake)
    dis_loss = - gen_loss

    beta = 0.1
    alpha = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
    interpolates = alpha*f_real + ((1.0-alpha)*f_fake)
    disc_interpolates = model.fnn(interpolates, W_d, b_d, act=tf.nn.leaky_relu)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    
    slopes = tf.norm(gradients, axis=1)
    gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
    
    dis_loss += beta*gradient_penalty
    
    dis_train = tf.train.AdamOptimizer(learning_rate=1.0e-4, beta1=0.5, beta2=0.9).minimize(dis_loss, var_list=var_list_dis)
    gen_train = tf.train.AdamOptimizer(learning_rate=1.0e-4, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=var_list_gen)

    saver_g = tf.train.Saver([weight for weight in W_g_z+b_g_z+W_g_x+b_g_x], max_to_keep=200)
    saver_d = tf.train.Saver([weight for weight in W_d+b_d])
    
    sess.run(tf.global_variables_initializer())
    
    n = 0
    nmax = 100000
    critic = 5
    _t0 = time.time()
    t0 = time.time()
    while n <= nmax:
        for i in range(critic):
            t_train, f_train = dataset.minibatch()
            dis_dict = {f_real: f_train.T}
            dis_loss_, dis_train_ = sess.run([dis_loss, dis_train], feed_dict=dis_dict)

        t_train, f_train = dataset.minibatch()
        gen_dict = {f_real: f_train.T}
        gen_loss_, gen_train_ = sess.run([gen_loss, gen_train], feed_dict=gen_dict)

        if n%100 == 0:
            print('Step: %d, Gen_loss: %.3e, Dis_loss: %.3e'%(n, dis_loss_, gen_loss_))
            t1 = time.time()
            print("Elapsed: ", t1 - t0)
            t0 = time.time()

        if n%10000 == 0:
            filename = './checkpoints/prior_' + str(int(idx)) + '_' + str(n)
            saver_g.save(sess, filename)
        n += 1
    _t1 = time.time()
    print("Elapsed: ", _t1 - _t0)
    np.savetxt("./outputs/time_"+str(int(idx))+".txt", np.array([_t1 - _t0]))
    saver_g.save(sess, './checkpoints/prior_'+str(int(idx)))
    saver_d.save(sess, './checkpoints/Dis')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.idx)
