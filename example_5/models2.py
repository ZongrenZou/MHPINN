import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time


tfd = tfp.distributions
tfb = tfp.bijectors


def jvp(y, x, v):
    # For more information, see https://github.com/renmengye/tensorflow-forward-ad/issues/2
    u = tf.ones_like(y) # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class PINN(tf.keras.Model):
    """PINN without boundary hard-encoded."""

    def __init__(self, name="pinn"):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(1),
            ]
        )
        self.nn.build(input_shape=[None, 2])
        self._name = name
        self.opt = tf.keras.optimizers.Adam(1e-3)

    def call(self, x, y):
        return self.nn(tf.concat([x, y], axis=-1))
    
    def pde(self, x, y):
        with tf.GradientTape() as g_xx, tf.GradientTape() as g_yy:
            g_xx.watch(x)
            g_yy.watch(y)
            with tf.GradientTape() as g_x, tf.GradientTape() as g_y:
                g_x.watch(x)
                g_y.watch(y)
                u = self.call(x, y)
            u_x = g_x.gradient(u, x)
            u_y = g_y.gradient(u, y)
        u_xx = g_xx.gradient(u_x, x)
        u_yy = g_yy.gradient(u_y, y)
        return u - u_xx - u_yy

    def loss_pde(self, x, y, f):
        f_pred = self.pde(x, y)
        return tf.reduce_mean((f_pred - f)**2)
    
    def loss_b(self, x, y):
        u_pred = self.call(x, y)
        return tf.reduce_mean(u_pred ** 2)

    def train_op(self, x, y, f, x_b, y_b):
        with tf.GradientTape() as tape:
            loss = self.loss_pde(x, y, f) + self.loss_b(x_b, y_b)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def train(self, x, y, f, x_b, y_b, niter=10000):
        x = tf.constant(x, tf.float32)
        y = tf.constant(y, tf.float32)
        f = tf.constant(f, tf.float32)
        x_b = tf.constant(x_b, tf.float32)
        y_b = tf.constant(y_b, tf.float32)
        train_op = tf.function(self.train_op)
        min_loss = 100

        for it in range(niter):
            loss = train_op(x, y, f, x_b, y_b)

            if it % 1000 == 0:
                print(it, loss.numpy())
                if loss.numpy() < min_loss:
                    min_loss = loss.numpy()
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class MHPINN(tf.keras.Model):
    """Multi-head PINN, without hard-encoded boundary condition."""

    def __init__(self, num_tasks=1000, dim=100, name="mhpinn"):
        super().__init__()
        self.shared_nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
            ]
        )
        self.dim = dim
        self.N = num_tasks
        self.heads = tf.Variable(0.05 * tf.random.normal(shape=[dim+1, self.N]), dtype=tf.float32)
        self.shared_nn.build(input_shape=[None, 2])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, y, heads):
        shared = self.shared_nn.call(tf.concat([x, y], axis=-1))
        out = tf.matmul(shared, heads[:self.dim, :]) + heads[self.dim, :]
        return out

    @tf.function
    def pde(self, x, y, heads):
        u = self.call(x, y, heads)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return u - u_xx - u_yy

    def loss_pde(self, x, y, f, heads):
        u = self.call(x, y, heads)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return tf.reduce_mean((u - u_xx - u_yy - f) ** 2)
    
    def loss_b(self, x, y, heads):
        return tf.reduce_mean(self.call(x, y, heads) ** 2)

    @tf.function
    def train_op(self, x_f, y_f, f, x_b, y_b):
        with tf.GradientTape() as tape:
            loss = self.loss_pde(x_f, y_f, f, self.heads) + self.loss_b(x_b, y_b)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x_f, y_f, f, x_b, y_b, niter=10000, ftol=5e-5):
        x_f = tf.constant(x_f, tf.float32)
        y_f = tf.constant(y_f, tf.float32)
        f = tf.constant(f, tf.float32)
        x_b = tf.constant(x_b, tf.float32)
        y_b = tf.constant(y_b, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x_f, y_f, f, x_b, y_b)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1], ", time: ", time.time() - t0)
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)