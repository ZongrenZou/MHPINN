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

    def __init__(self, eps=0, name="pinn"):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(100, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(100, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(100, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
            ]
        )
        self.nn.build(input_shape=[None, 2])
        self.opt = tf.keras.optimizers.Adam(1e-3)
        self.eps = eps
        self._name = name

    def call(self, x, y):
        return x * (1 - x) * y * (1 - y) * self.nn(tf.concat([x, y], axis=-1))

    def pde(self, x, y, f):
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
        f_pred = 0.1 * (u_xx + u_yy) + u * (u ** 2 - 1)
        return tf.reduce_mean((f_pred - f) ** 2)
    
    @tf.function
    def f(self, x, y):
        u = self.call(x, y)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        return 0.1 * (u_xx + u_yy) + u * (u ** 2 - 1)

    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.pde(x, y, f) ** 2) + self.eps * tf.reduce_sum(self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def train(self, x, y, f, niter=10000):
        x = tf.constant(x, tf.float32)
        y = tf.constant(y, tf.float32)
        f = tf.constant(f, tf.float32)
        train_op = tf.function(self.train_op)
        min_loss = 100

        for it in range(niter):
            loss = train_op(x, y, f)

            if it % 1000 == 0:
                print(it, loss.numpy())
                if loss.numpy() < min_loss:
                    min_loss = loss.numpy()
                    self.save_weights(
                        filepath="./checkpoints/" + self.name,
                        overwrite=True,
                    )

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)


class MHPINN(tf.keras.Model):

    def __init__(self, num_tasks=1000, dim=100, eps=0, name="mhpinn"):
        super().__init__()
        self.shared_nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
            ]
        )
        self.dim = dim
        self.eps = eps
        self.N = num_tasks
        self.heads = tf.Variable(0.05 * tf.random.normal(shape=[dim+1, self.N]), dtype=tf.float32)
        self.shared_nn.build(input_shape=[None, 2])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, y, heads):
        shared = self.shared_nn.call(tf.concat([x, y], axis=-1))
        out = tf.matmul(shared, heads[:self.dim, :]) + heads[self.dim:, :] # shape of [batch_size, self.N]
        return x * (1-x) * y * (1-y) * out

    @tf.function
    def pde(self, x, y, heads):
        u = self.call(x, y, heads)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        f_pred = 0.1 * (u_xx + u_yy) + u * (u**2  - 1)
        return f_pred

    def loss_function(self, x, y, f):
        u = self.call(x, y, self.heads)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        f_pred = 0.1 * (u_xx + u_yy) + u * (u**2  - 1)
        return tf.reduce_mean((f_pred - f) ** 2)

    @tf.function
    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            # regularization = tf.reduce_mean(self.heads ** 2)
            loss = self.loss_function(x, y, f)
            total_loss = loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss, loss

    def train(self, x_train, y_train, f_train, niter=10000, ftol=5e-5):
        x_train = tf.constant(x_train, tf.float32)
        y_train = tf.constant(y_train, tf.float32)
        f_train = tf.constant(f_train, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            total_loss, loss_value = train_op(x_train, y_train, f_train)
            loss += [loss_value.numpy()]
            if (it + 1) % 1000 == 0:
                print(it, loss[-1], total_loss.numpy(), ", time: ", time.time() - t0)
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/" + self.name,
                        overwrite=True,
                    )
                t0 = time.time()
                if loss[-1] < ftol:
                    break
        return loss

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)

    # def call_batch(self, x, y, idx):
    #     shared = self.shared_nn.call(tf.concat([x, y], axis=-1))
    #     xi = tf.transpose(tf.gather(self.xi, idx, axis=0))
    #     out = tf.matmul(shared, xi[:self.dim, :]) + xi[self.dim:, :] # shape of [batch_size, self.N]
    #     return x * (1-x) * y * (1-y) * out

    # def loss_function_batch(self, x, y, f, idx):
    #     u = self.call_batch(x, y, idx)
    #     v = tf.ones_like(x)
    #     u_x = jvp(u, x, v)[0]
    #     u_xx = jvp(u_x, x, v)[0]
    #     u_y = jvp(u, y, v)[0]
    #     u_yy = jvp(u_y, y, v)[0]
    #     return tf.reduce_mean((- u_xx - u_yy - f) ** 2)

    # @tf.function
    # def train_batch_op(self, x, y, f, idx):
    #     with tf.GradientTape() as tape:
    #         # regularization = tf.reduce_mean(self.xi ** 2)
    #         loss = self.loss_function_batch(x, y, f, idx)
    #         total_loss = loss #+ self.eps * regularization
    #     grads = tape.gradient(total_loss, self.trainable_variables)
    #     self.opt.apply_gradients(zip(grads, self.trainable_variables))
    #     return total_loss, loss

    # def train_batch(self, x_train, y_train, f_train, batch_size=100, nepoch=10000, ftol=5e-5):
    #     x_train = tf.constant(x_train, tf.float32)
    #     y_train = tf.constant(y_train, tf.float32)
    #     f_train = tf.constant(f_train, tf.float32)

    #     train_batch_op = tf.function(self.train_batch_op)
    #     loss_op = tf.function(self.loss_function)
    #     loss = []
    #     min_loss = 1000
    #     N = f_train.shape[1]

    #     t0 = time.time()
    #     for epoch in range(nepoch):
    #         idx = np.random.choice(N, N, replace=False)
    #         for i in range(N // batch_size):
    #             # x_batch = tf.gather(x_train, idx[i*batch_size:(i+1)*batch_size], axis=0)
    #             # y_batch = tf.gather(y_train, idx[i*batch_size:(i+1)*batch_size], axis=0)
    #             idx_batch = tf.constant(idx[i*batch_size:(i+1)*batch_size], tf.int32)
    #             f_batch = tf.gather(f_train, idx_batch, axis=1)
    #             _ = train_batch_op(x_train, y_train, f_batch, idx_batch)
    #         loss_value = loss_op(x_train, y_train, f_train).numpy()
    #         if loss_value < ftol:
    #             break
    #         print(epoch, loss_value, ", time: ", time.time() - t0)
    #         if loss_value < min_loss:
    #             min_loss = loss_value
    #             self.save_weights(
    #                 filepath="./checkpoints/min_loss",
    #                 overwrite=True,
    #             )
    #         t0 = time.time()


class Downstream(tf.keras.Model):

    def __init__(self, mhnn, flow, dim, eps=0.1, name="downstream"):
        super().__init__()
        self.body = mhnn.shared_nn
        self.log_prob_fn = tf.function(flow.log_prob)
        self.sample_fn = tf.function(flow.sample)
        self.dim = dim
        
        init = self.sample_fn([1])
        self.head = tf.Variable(tf.transpose(init))
        self.eps = eps
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, y):
        basis = self.body(tf.concat([x, y], axis=-1))
        out = tf.matmul(basis, self.head[:self.dim]) + self.head[self.dim:]
        return x * (1-x) * y * (1-y) * out

    @tf.function
    def pde(self, x, y):
        u = self.call(x, y)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        f = 0.1 * (u_xx + u_yy) + u * (u ** 2 - 1)
        return f

    def loss_function(self, x, y, f):
        u = self.call(x, y)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        f_pred = 0.1 * (u_xx + u_yy) + u * (u ** 2 - 1)
        return tf.reduce_mean((f_pred - f) ** 2)
    
    @tf.function
    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            regularization = - tf.math.reduce_sum(self.log_prob_fn(tf.transpose(self.head)))
            losses = self.loss_function(x, y, f) 
            total_loss = tf.reduce_sum(losses) + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return total_loss

    def train(self, x, y, f, niter=10000):
        x = tf.constant(x, tf.float32)
        y = tf.constant(y, tf.float32)
        f = tf.constant(f, tf.float32)

        train_op = self.train_op
        loss_op = tf.function(lambda : self.loss_function(x, y, f))

        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(x, y, f)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/" + self.name,
                        overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)
        