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


class MHPINN(tf.keras.Model):
    """Multi-head PINN."""
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
        self.log_ks = tf.Variable(
            tf.zeros(shape=[1, self.N]), dtype=tf.float32,
        )
        self.shared_nn.build(input_shape=[None, 1])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.task_eps = tf.ones(shape=[num_tasks])

    def call(self, inputs, heads):
        shared = self.shared_nn.call(inputs)  # shape of [batch_size, 100]
        out = (
            tf.matmul(shared, heads[: self.dim, :]) + heads[self.dim :, :]
        )  # shape of [batch_size, self.N]
        return inputs ** 2 * out
        # return out

    @tf.function
    def pde(self, t, heads, log_ks):
        u = self.call(t, heads)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u_tt = jvp(u_t, t, v)[0]
        f_pred = u_tt + tf.math.exp(log_ks) * tf.math.sin(u)
        return f_pred

    def loss_function(self, t_f, f, t_u, u):
        u_pred = self.call(t_f, self.heads)
        v = tf.ones_like(t_f)
        u_t_pred = jvp(u_pred, t_f, v)[0]
        u_tt_pred = jvp(u_t_pred, t_f, v)[0]
        f_pred = u_tt_pred + tf.math.exp(self.log_ks) * tf.math.sin(u_pred)
        loss_f = tf.reduce_mean((f_pred - f) ** 2)

        u_pred = self.call(t_u, self.heads)
        loss_u = tf.reduce_mean(self.task_eps * (u_pred - u) ** 2)

        return loss_f + 10 * loss_u

    @tf.function
    def train_op(self, t_f, f, t_u, u):
        with tf.GradientTape() as tape:
            loss = self.loss_function(t_f, f, t_u, u)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t_f, f, t_u, u, niter=10000):
        t_f = tf.constant(t_f, tf.float32)
        f = tf.constant(f, tf.float32)
        t_u = tf.constant(t_u, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = self.train_op
        loss_op = tf.function(lambda : self.loss_function(t_f, f, t_u, u))
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(t_f, f, t_u, u)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss, ", time: ", time.time() - t0)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/" + self.name, overwrite=True,
                    )
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)


class Model(tf.keras.Model):
    """Model with shared body and prior distribution of the heads."""
    def __init__(self, body, flow, dim, eps=0.1):
        super().__init__()
        self.body = body
        self.log_prob_fn = flow.log_prob
        self.sample_fn = flow.sample
        self.dim = dim

        init = self.sample_fn([1])
        init = tf.zeros_like(init)
        self.head = tf.Variable(tf.transpose(init[:, :-1]))
        self.log_k = tf.Variable(tf.transpose(init[:, -1:]))
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        shared = self.body.shared_nn(inputs)
        out = (
            tf.matmul(shared, self.head[: self.dim, :]) + self.head[self.dim :, :]
        )
        return inputs ** 2 * out

    @tf.function
    def pde(self, t):
        u = self.call(t)
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        f_pred = u_tt + tf.math.exp(self.log_k) * tf.math.sin(u)
        return f_pred

    def loss_function(self, t_f, f, t_u, u):
        u_pred = self.call(t_f)
        v = tf.ones_like(t_f)
        u_t_pred = jvp(u_pred, t_f, v)[0]
        u_tt_pred = jvp(u_t_pred, t_f, v)[0]
        # u_t_pred = tf.gradients(u_pred, t_f)[0]
        # u_tt_pred = tf.gradients(u_t_pred, t_f)[0]
        f_pred = u_tt_pred + tf.math.exp(self.log_k) * tf.math.sin(u_pred)
        loss_f = tf.reduce_mean((f_pred - f) ** 2)

        u_pred = self.call(t_u)
        loss_u = tf.reduce_mean((u_pred - u) ** 2)

        return loss_f + 10 * loss_u

    @tf.function
    def train_op(self, t_f, f, t_u, u):
        with tf.GradientTape() as tape:
            s = tf.concat([self.head, self.log_k], axis=0)
            regularization = -tf.math.reduce_sum(
                self.log_prob_fn(tf.transpose(s)),
            )
            total_loss = self.loss_function(t_f, f, t_u, u) + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head, self.log_k])
        self.opt.apply_gradients(zip(grads, [self.head, self.log_k]))
        return total_loss

    def train(self, t_f, f, t_u, u, niter=10000):
        t_f = tf.constant(t_f, tf.float32)
        f = tf.constant(f, tf.float32)
        t_u = tf.constant(t_u, tf.float32)
        u = tf.constant(u, tf.float32)
        train_op = self.train_op

        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(t_f, f, t_u, u)
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
        self.log_k = tf.Variable(0.0, dtype=tf.float32)

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        return inputs ** 2 * self.nn.call(inputs)
        # return self.nn.call(inputs)

    def loss_function(self, t_f, f, t_u, u):
        u_pred = self.call(t_f)
        v = tf.ones_like(t_f)
        u_t_pred = jvp(u_pred, t_f, v)[0]
        u_tt_pred = jvp(u_t_pred, t_f, v)[0]
        k = tf.math.exp(self.log_k)
        f_pred = u_tt_pred + k * tf.math.sin(u_pred)
        loss_f = tf.reduce_mean((f_pred - f) ** 2)

        u_pred = self.call(t_u)
        loss_u = tf.reduce_mean((u_pred - u) ** 2)
        return loss_f + 10 * loss_u

    @tf.function
    def pde(self, t):
        u = self.call(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u_tt = jvp(u_t, t, v)[0]
        k = tf.math.exp(self.log_k)
        f_pred = u_tt + k * tf.math.sin(u)
        return f_pred

    def train_op(self, t_f, f, t_u, u):
        with tf.autodiff.GradientTape() as tape:
            loss = self.loss_function(t_f, f, t_u, u) + tf.reduce_sum(self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t_f, f, t_u, u, niter=10000):
        t_f = tf.constant(t_f, tf.float32)
        f = tf.constant(f, tf.float32)
        t_u = tf.constant(t_u, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = tf.function(self.train_op)
        # train_op = self.train_op
        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(t_f, f, t_u, u)
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
