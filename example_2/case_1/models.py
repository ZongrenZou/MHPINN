import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time


tfd = tfp.distributions
tfb = tfp.bijectors


def jvp(y, x, v):
    # For more information, see https://github.com/renmengye/tensorflow-forward-ad/issues/2
    u = tf.ones_like(y)  # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


# class PINN(tf.keras.Model):

#     def __init__(self, eps=0.1):
#         super().__init__()
#         self.nn = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(eps)),
#                 tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(eps)),
#                 tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(eps)),
#                 tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.L2(eps)),
#             ]
#         )
#         self.nn.build(input_shape=[None, 1])

#         self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

#     def call(self, inputs):
#         return inputs * self.nn.call(inputs)

#     def loss_function(self, t, f):
#         k = 1
#         u = self.call(t)
#         u1, u2 = tf.split(u, 2, axis=-1)

#         v = tf.ones_like(t)
#         u1_t = jvp(u1, t, v)[0]
#         u2_t = jvp(u2, t, v)[0]
#         loss = tf.reduce_mean((u1_t - u2) ** 2) + tf.reduce_mean((u2_t + k*tf.math.sin(u1) - f) ** 2)
#         return loss

#     @tf.function
#     def pde(self, t):
#         k = 1
#         u = self.call(t)
#         u1, u2 = tf.split(u, 2, axis=-1)

#         v = tf.ones_like(t)
#         u2_t = jvp(u2, t, v)[0]

#         f_pred = u2_t + k*tf.math.sin(u1)
#         return f_pred

#     def train_op(self, inputs, targets):
#         with tf.autodiff.GradientTape() as tape:
#             loss = self.loss_function(inputs, targets) + tf.reduce_sum(self.losses)
#         grads = tape.gradient(loss, self.trainable_variables)
#         self.opt.apply_gradients(zip(grads, self.trainable_variables))
#         return loss

#     def train(self, inputs, targets, niter=10000):
#         inputs_train = tf.constant(inputs, tf.float32)
#         targets_train = tf.constant(targets, tf.float32)
#         train_op = tf.function(self.train_op)
#         # train_op = self.train_op
#         min_loss = 1000
#         loss = []

#         for it in range(niter):
#             loss_value = train_op(inputs_train, targets_train)
#             loss += [loss_value.numpy()]
#             if it % 1000 == 0:
#                 print(it, loss[-1])
#                 if loss[-1] < min_loss:
#                     min_loss = loss[-1]
#                     self.save_weights(
#                         filepath="./checkpoints/single",
#                         overwrite=True,
#                     )

#         return loss

#     def restore(self):
#         self.load_weights("./checkpoints/single")


class PINN(tf.keras.Model):
    def __init__(self, eps=0.1):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    50,
                    activation=tf.tanh,
                    kernel_regularizer=tf.keras.regularizers.L2(eps),
                ),
                tf.keras.layers.Dense(
                    50,
                    activation=tf.tanh,
                    kernel_regularizer=tf.keras.regularizers.L2(eps),
                ),
                tf.keras.layers.Dense(
                    50,
                    activation=tf.tanh,
                    kernel_regularizer=tf.keras.regularizers.L2(eps),
                ),
                tf.keras.layers.Dense(
                    1, kernel_regularizer=tf.keras.regularizers.L2(eps)
                ),
            ]
        )
        self.nn.build(input_shape=[None, 1])

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        return inputs ** 2 * self.nn.call(inputs)

    def loss_function(self, t, f):
        k = 1
        u = self.call(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u_tt = jvp(u_t, t, v)[0]
        f_pred = u_tt + k * tf.math.sin(u)
        return tf.reduce_mean((f_pred - f) ** 2)

    @tf.function
    def pde(self, t):
        k = 1
        u = self.call(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u_tt = jvp(u_t, t, v)[0]
        f_pred = u_tt + k * tf.math.sin(u)
        return f_pred

    def train_op(self, inputs, targets):
        with tf.autodiff.GradientTape() as tape:
            loss = self.loss_function(inputs, targets) + tf.reduce_sum(self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, inputs, targets, niter=10000):
        inputs_train = tf.constant(inputs, tf.float32)
        targets_train = tf.constant(targets, tf.float32)
        train_op = tf.function(self.train_op)
        # train_op = self.train_op
        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(inputs_train, targets_train)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1])
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/single", overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/single")


class Meta(tf.keras.Model):
    def __init__(self, num_tasks=1000, dim=50, name="meta"):
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
        self.heads = tf.Variable(
            0.05 * tf.random.normal(shape=[dim + 1, self.N]), dtype=tf.float32
        )
        self.shared_nn.build(input_shape=[None, 1])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs, heads):
        shared = self.shared_nn.call(inputs)  # shape of [batch_size, 100]
        out = (
            tf.matmul(shared, heads[: self.dim, :]) + heads[self.dim :, :]
        )  # shape of [batch_size, self.N]
        return inputs ** 2 * out

    @tf.function
    def pde(self, t, heads):
        k = 1
        u = self.call(t, heads)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u_tt = jvp(u_t, t, v)[0]
        f_pred = u_tt + k * tf.math.sin(u)
        return f_pred

    def loss_function(self, t, f):
        k = 1
        u = self.call(t, self.heads)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u_tt = jvp(u_t, t, v)[0]
        f_pred = u_tt + k * tf.math.sin(u)
        return tf.reduce_mean((f_pred - f) ** 2)

    @tf.function
    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = self.loss_function(inputs, targets)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, inputs, targets, niter=10000, ftol=5e-6):
        inputs_train = tf.constant(inputs, tf.float32)
        targets_train = tf.constant(targets, tf.float32)
        # train_op = tf.function(self.train_op)
        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(inputs_train, targets_train)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1], ", time: ", time.time() - t0)
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/" + self.name, overwrite=True,
                    )
                if loss[-1] < ftol:
                    break
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)


class Model(tf.keras.Model):
    def __init__(self, body, flow, dim, eps=0.1):
        super().__init__()
        self.body = body
        self.log_prob_fn = tf.function(flow.log_prob)
        self.sample_fn = tf.function(flow.sample)
        self.dim = dim

        init = self.sample_fn([1])
        self.head = tf.Variable(tf.transpose(init))
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        basis = self.body.shared_nn(inputs)
        return (inputs ** 2 - 1) * (
            tf.matmul(basis, self.head[: self.dim]) + self.head[self.dim :]
        )

    @tf.function
    def pde(self, x):
        u = self.call(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_pred = 0.01 * u_xx - u ** 3
        return f_pred

    def loss_function(self, x, f):
        u = self.call(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_pred = 0.01 * u_xx - u ** 3
        return tf.reduce_mean((f - f_pred) ** 2)

    @tf.function
    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            regularization = -tf.math.reduce_sum(
                self.log_prob_fn(tf.transpose(self.head))
            )
            total_loss = self.loss_function(inputs, targets) + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return total_loss

    def train(self, inputs, targets, niter=10000):
        inputs_train = tf.constant(inputs, tf.float32)
        targets_train = tf.constant(targets, tf.float32)
        train_op = self.train_op

        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(inputs_train, targets_train)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1])
                if loss_value < min_loss and it > niter // 2:
                    min_loss = loss_value
                    self.save_weights(
                        filepath="./checkpoints/model", overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/model")


class Model2(tf.keras.Model):
    """Model for downstream tasks with L2 regularization"""

    def __init__(self, body, dim, eps=0.0):
        super().__init__()
        self.body = body
        self.dim = dim
        self.head = tf.Variable(
            0.05 * tf.random.normal(shape=[dim + 1, 1]), dtype=tf.float32
        )
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        basis = self.body.shared_nn(inputs)
        return (inputs ** 2 - 1) * (
            tf.matmul(basis, self.head[: self.dim]) + self.head[self.dim :]
        )

    @tf.function
    def pde(self, x):
        u = self.call(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_pred = 0.01 * u_xx - u ** 3
        return f_pred

    def loss_function(self, x, f):
        u = self.call(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_pred = 0.01 * u_xx - u ** 3
        return tf.reduce_mean((f - f_pred) ** 2)

    @tf.function
    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            regularization = tf.math.reduce_sum(self.head ** 2)
            total_loss = self.loss_function(inputs, targets) + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return total_loss

    def train(self, inputs, targets, niter=10000):
        inputs_train = tf.constant(inputs, tf.float32)
        targets_train = tf.constant(targets, tf.float32)
        train_op = self.train_op

        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(inputs_train, targets_train)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1])
                if loss_value < min_loss and it > niter // 2:
                    min_loss = loss_value
                    self.save_weights(
                        filepath="./checkpoints/model2", overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/model2")


class LA(tf.keras.Model):
    """Model for downstream tasks, with Laplace approximation."""

    def __init__(self, body, flow, dim, noise):
        super().__init__()
        self.body = body
        self.flow = flow
        self.log_prob_fn = tf.function(flow.log_prob)
        self.dim = dim
        self.head = tf.Variable(tf.transpose(flow.sample([1])), dtype=tf.float32)

        self.dist = tfp.distributions.Normal(loc=0, scale=noise)

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        basis = self.body.shared_nn(inputs)
        return (inputs ** 2 - 1) * (
            tf.matmul(basis, self.head[: self.dim]) + self.head[self.dim :]
        )

    @tf.function
    def pde(self, x):
        u = self.call(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_pred = 0.01 * u_xx - u ** 3
        return f_pred

    def loss_function(self, x, f):
        u = self.call(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f_pred = 0.01 * u_xx - u ** 3
        return tf.reduce_mean((f - f_pred) ** 2)

    def neg_log_posterior(self, inputs, targets):
        u = self.call(inputs)
        u_x = tf.gradients(u, inputs)[0]
        u_xx = tf.gradients(u_x, inputs)[0]
        f_pred = 0.01 * u_xx - u ** 3

        log_prior = tf.reduce_sum(self.log_prob_fn(tf.transpose(self.head)))
        log_likelihood = tf.reduce_sum(self.dist.log_prob(f_pred - targets))
        return -log_prior - log_likelihood

    @tf.function
    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            neg_log_posterior = self.neg_log_posterior(inputs, targets)
        grads = tape.gradient(neg_log_posterior, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return neg_log_posterior

    def train(self, inputs, targets, niter=10000):
        inputs_train = tf.constant(inputs, tf.float32)
        targets_train = tf.constant(targets, tf.float32)
        train_op = self.train_op

        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(inputs_train, targets_train)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1])
                if loss_value < min_loss and it > niter // 2:
                    min_loss = loss_value
                    self.save_weights(
                        filepath="./checkpoints/la", overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/la")


# class GAN(tf.keras.Model):

#     def __init__(
#         generator_layers_z,
#         generator_layers_x,
#         discriminator_layers,
#     ):
#         super().__init__()
#         denses_x = []
#         denses_z = []
#         denses_D = []

#         for i in range(len(generator_layers_x) - 1):
