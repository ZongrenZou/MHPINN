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

    def __init__(self, eps=0.1, ws=[1, 1]):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
                tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
                tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
                tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
            ]
        )
        self.nn.build(input_shape=[None, 2])
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.ws = ws

    def call(self, inputs):
        return (inputs[:, 0:1] ** 2 - 1) * self.nn.call(inputs)

    def loss_function(self, inputs_f, inputs_0, targets_0):
        pred_0 = self.call(inputs_0)
        loss_init = tf.reduce_mean((pred_0 - targets_0) ** 2)

        u = self.call(inputs_f)
        u_xt = tf.gradients(u, inputs_f)[0]
        u_x, u_t = tf.split(u_xt, 2, axis=-1)
        u_xx = tf.gradients(u_x, inputs_f)[0][:, 0:1]
        f_pred = u_t - (0.1 * u_xx + 0.1 * u * (1-u))
        loss_f = tf.reduce_mean(f_pred ** 2)
        return loss_f, loss_init

    def train_op(self, inputs_f, inputs_0, targets_0):
        with tf.autodiff.GradientTape() as tape:
            loss_f, loss_init = self.loss_function(
                inputs_f, inputs_0, targets_0,
            )
            loss = self.ws[0] * loss_f + self.ws[1] * loss_init
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, inputs_f, inputs_0, targets_0, niter=10000):
        inputs_f = tf.constant(inputs_f, tf.float32)
        inputs_0 = tf.constant(inputs_0, tf.float32)
        targets_0 = tf.constant(targets_0, tf.float32)

        train_op = tf.function(self.train_op)
        # train_op = self.train_op
        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(inputs_f, inputs_0, targets_0)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1])
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/single",
                        overwrite=True,
                    )

        return loss
    
    def restore(self):
        self.load_weights("./checkpoints/single")


class MHPINN(tf.keras.Model):

    def __init__(self, num_tasks=1000, dim=50, ws=[1, 1], name="meta"):
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
        self.ws = ws

        self.pde_fn = tf.function(self._pde)
        self.train_fn = tf.function(self._train_fn)

    def call(self, inputs, heads):
        shared = self.shared_nn.call(inputs) # shape of [batch_size, 100]
        out = tf.matmul(shared, heads[:self.dim, :]) + heads[self.dim:, :] # shape of [batch_size, self.N]
        return (inputs[:, 0:1] ** 2 - 1) * out
    
    def _pde(self, inputs, heads):
        x, t = tf.split(inputs, 2, axis=-1)
        u = self.call(tf.concat([x, t], axis=-1), heads)

        u_x = jvp(u, x, tf.ones_like(x))[0]
        u_xx = jvp(u_x, x, tf.ones_like(x))[0]
        u_t = jvp(u, t, tf.ones_like(t))[0]
        
        return u_t - (0.1 * u_xx + 0.1 * u * (1 - u))

    def loss_function(self, inputs_f, inputs_0, targets_0):
        f_pred = self._pde(inputs_f, self.heads)
        u0_pred = self.call(inputs_0, self.heads)
        loss_f = tf.reduce_mean(f_pred ** 2)
        loss_0 = tf.reduce_mean((u0_pred - targets_0) ** 2)
        return loss_f, loss_0

    def _train_fn(self, inputs_f, inputs_0, targets_0):
        with tf.GradientTape() as tape:
            losses = self.loss_function(inputs_f, inputs_0, targets_0)
            loss = self.ws[0] * losses[0] + self.ws[1] * losses[1]
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, losses

    def train(self, inputs_f, inputs_0, targets_0, niter=10000, ftol=5e-6):
        inputs_f = tf.constant(inputs_f, tf.float32)
        inputs_0 = tf.constant(inputs_0, tf.float32)
        targets_0 = tf.constant(targets_0, tf.float32)

        def _loss_op():
            losses = self.loss_function(inputs_f, inputs_0, targets_0)
            return self.ws[0] * losses[0] + self.ws[1] * losses[1]
        loss_op = tf.function(_loss_op)
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value, losses = self.train_fn(inputs_f, inputs_0, targets_0)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss, ", time: ", time.time() - t0)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
                if loss[-1] < ftol:
                    break
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class Downstream(tf.keras.Model):

    def __init__(self, body, flow, dim, ws=[1, 1], eps=0.1):
        super().__init__()
        self.body = body
        self.log_prob_fn = tf.function(flow.log_prob)
        self.sample_fn = tf.function(flow.sample)
        self.dim = dim
        
        init = self.sample_fn([1])
        # init = 0.05 * tf.random.normal(shape=[1, 51])
        self.head = tf.Variable(tf.transpose(init))
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.ws = ws
    
    def call(self, inputs):
        basis = self.body.shared_nn(inputs)
        return (inputs[:, 0:1] ** 2 - 1) * (tf.matmul(basis, self.head[:self.dim]) + self.head[self.dim:])

    def loss_function(self, inputs_f, inputs_0, targets_0):
        pred_0 = self.call(inputs_0)
        loss_init = tf.reduce_mean((pred_0 - targets_0) ** 2)

        u = self.call(inputs_f)
        u_xt = tf.gradients(u, inputs_f)[0]
        u_x, u_t = tf.split(u_xt, 2, axis=-1)
        u_xx = tf.gradients(u_x, inputs_f)[0][:, 0:1]
        f_pred = u_t - (0.1 * u_xx + 0.1 * u * (1 - u))
        loss_f = tf.reduce_mean(f_pred ** 2)
        return loss_f, loss_init
    
    @tf.function
    def train_op(self, inputs_f, inputs_0, targets_0):
        with tf.GradientTape() as tape:
            regularization = - tf.math.reduce_sum(self.log_prob_fn(tf.transpose(self.head)))
            losses = self.loss_function(inputs_f, inputs_0, targets_0) 
            total_loss = self.ws[0] * losses[0] + self.ws[1] * losses[1] + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return total_loss

    def train(self, inputs_f, inputs_0, targets_0, niter=10000):
        inputs_f = tf.constant(inputs_f, tf.float32)
        inputs_0 = tf.constant(inputs_0, tf.float32)
        targets_0 = tf.constant(targets_0, tf.float32)

        train_op = self.train_op

        min_loss = 1000
        loss = []

        for it in range(niter):
            loss_value = train_op(inputs_f, inputs_0, targets_0)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1])
                if loss_value < min_loss and it > niter//2:
                    min_loss = loss_value
                    self.save_weights(
                        filepath="./checkpoints/model",
                        overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/model")


class LA(tf.keras.Model):
    """Model for downstream tasks, with Laplace approximation."""

    def __init__(self, body, flow, dim, noise_f, noise_u):
        super().__init__()
        self.body = body
        self.log_prob_fn = tf.function(flow.log_prob)
        self.sample_fn = tf.function(flow.sample)
        self.dim = dim
        
        #init = self.sample_fn([1])
        init = 0.05 * tf.random.normal(shape=[1, 51])
        self.head = tf.Variable(tf.transpose(init))
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.dist_f = tfp.distributions.Normal(loc=0, scale=noise_f)
        self.dist_u = tfp.distributions.Normal(loc=0, scale=noise_u)
    
    def call(self, inputs):
        basis = self.body.shared_nn(inputs)
        return (inputs[:, 0:1] ** 2 - 1) * (tf.matmul(basis, self.head[:self.dim]) + self.head[self.dim:])

    def loss_function(self, inputs_f, inputs_0, targets_0):
        pred_0 = self.call(inputs_0)
        loss_init = tf.reduce_mean((pred_0 - targets_0) ** 2)

        u = self.call(inputs_f)
        u_xt = tf.gradients(u, inputs_f)[0]
        u_x, u_t = tf.split(u_xt, 2, axis=-1)
        u_xx = tf.gradients(u_x, inputs_f)[0][:, 0:1]
        f_pred = u_t - (0.1 * u_xx + 0.1 * u * (1 - u))
        loss_f = tf.reduce_mean(f_pred ** 2)
        return loss_f, loss_init

    def neg_log_posterior(self, inputs, targets):
        u = self.call(inputs)
        u_x = tf.gradients(u, inputs)[0]
        u_xx = tf.gradients(u_x, inputs)[0]
        f_pred = 0.01 * u_xx - u ** 3

        log_prior = tf.reduce_sum(self.log_prob_fn(tf.transpose(self.head)))
        log_likelihood = tf.reduce_sum(self.dist.log_prob(f_pred - targets))
        return - log_prior - log_likelihood

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
                if loss_value < min_loss and it > niter//2:
                    min_loss = loss_value
                    self.save_weights(
                        filepath="./checkpoints/la",
                        overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/la")