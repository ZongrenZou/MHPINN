import tensorflow.compat.v1 as tf
import numpy as np

class DNN:
    def __init__(self):
        pass

    def hyper_initial(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2.0/(in_dim+out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)

        return W, b

    #discriminator
    def fnn(self, X, W, b, act):
        A = X
        L = len(W)
        for i in range(L-1):
            A = act(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])

        return Y

    #pdenet
    def pdenn(self, t, u):
        k = 1
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        f = u_tt + k * tf.math.sin(u) 
        return f
