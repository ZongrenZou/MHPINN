import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time


tfd = tfp.distributions
tfb = tfp.bijectors


class MHNN(tf.keras.Model):
    """Multi-head NN."""

    def __init__(self, num_tasks=1000, dim=100, name="mhnn"):
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
        # fan_in = 50
        # fan_out = 1
        # limit = tf.math.sqrt(6/(fan_in+fan_out))
        # weights = - limit + 2 * limit * tf.random.uniform(shape=[dim, self.N])
        # bias = tf.zeros(shape=[1, self.N])
        # self.heads = tf.Variable(
        #     tf.concat([weights, bias], axis=0), dtype=tf.float32,
        # )
        self.heads = tf.Variable(0.05 * tf.random.normal(shape=[dim+1, self.N]), dtype=tf.float32)

        self.shared_nn.build(input_shape=[None, 1])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, heads):
        shared = self.shared_nn.call(x)
        out = tf.matmul(shared, heads[:self.dim, :]) + heads[self.dim, :]
        return out

    def loss_function(self, x, u):
        return tf.reduce_mean((self.call(x, self.heads) - u)**2)

    @tf.function
    def train_op(self, x, u):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x,u)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x, u, niter=10000, ftol=5e-5):
        x = tf.constant(x, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = self.train_op
        loss_op = tf.function(lambda : self.loss_function(x, u))
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x, u)
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
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class PINN(tf.keras.Model):

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
        self.log_lamb = tf.Variable(0.0, dtype=tf.float32)
        self.nn.build(input_shape=[None, 2])
        self._name = name
        self.opt = tf.keras.optimizers.Adam(1e-3)

    def call(self, x, y):
        out = self.nn(tf.concat([x, y], axis=-1))
        out = x * (2*np.pi-x) * y * (2*np.pi-y) * out 
        return out / np.pi ** 4

    def pde(self, x, y):
        lamb = tf.math.exp(self.log_lamb)
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
        return lamb ** 2 *u - u_xx - u_yy

    def loss_pde(self, x, y, f):
        lamb = tf.math.exp(self.log_lamb)
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
        return tf.reduce_mean((lamb ** 2 * u - u_xx - u_yy - f)**2)
    
    def loss_u(self, x, y, u):
        return tf.reduce_mean((self.call(x, y) - u)**2)

    def train_op(self, x_f, y_f, f, x_u, y_u, u):
        with tf.GradientTape() as tape:
            loss = self.loss_pde(x_f, y_f, f) + self.loss_u(x_u, y_u, u)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def train(self, x_f, y_f, f, x_u, y_u, u, niter=10000):
        x_f = tf.constant(x_f, tf.float32)
        y_f = tf.constant(y_f, tf.float32)
        f = tf.constant(f, tf.float32)
        x_u = tf.constant(x_u, tf.float32)
        y_u = tf.constant(y_u, tf.float32)
        u = tf.constant(u, tf.float32)
        train_op = tf.function(self.train_op)
        min_loss = 100

        for it in range(niter):
            loss = train_op(x_f, y_f, f, x_u, y_u, u)

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


class Downstream(tf.keras.Model):
    """For downstream tasks."""

    def __init__(self, mhnn, flow, dim, eps=0.1, name="downstream"):
        super().__init__()
        self.body = mhnn.shared_nn
        self.log_prob_fn = flow.log_prob
        self.sample_fn = flow.sample
        self.dim = dim

        init = self.sample_fn([1])

        self.head = tf.Variable(tf.transpose(init[:, :dim+1]), dtype=tf.float32)

        self.eps = eps
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        shared = self.body.call(x)
        out = tf.matmul(shared, self.head[:self.dim, :]) + self.head[self.dim, :]
        return out
    
    def loss_function(self, x, u):
        return tf.reduce_mean((self.call(x) - u) ** 2)

    @tf.function
    def train_op(self, x, u):
        with tf.GradientTape() as tape:
            regularization = - tf.math.reduce_sum(
                self.log_prob_fn(tf.transpose(self.head))
            )
            loss = self.loss_function(x, u) + \
                   self.eps * regularization
        grads = tape.gradient(loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)

    def train(self, x, u, niter=10000):
        x = tf.constant(x, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = self.train_op
        loss_op = tf.function(lambda : self.loss_function(x, u))
        loss = []
        min_loss = 1000

        for it in range(niter):
            loss_value = train_op(x, u)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss


class Reference(tf.keras.Model):
    """Reference model."""

    def __init__(self, mhnn, dim, eps=0.1, name="reference"):
        super().__init__()
        self.body = mhnn.shared_nn
        self.dim = dim

        self.head = tf.Variable(0.05 * tf.random.normal(shape=[dim+1, 1]), dtype=tf.float32)

        self.eps = eps
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        shared = self.body.call(x)
        out = tf.matmul(shared, self.head[:self.dim, :]) + self.head[self.dim, :]
        return out
    
    def loss_function(self, x, u):
        return tf.reduce_mean((self.call(x) - u) ** 2)

    @tf.function
    def train_op(self, x, u):
        with tf.GradientTape() as tape:
            # L2 regularization
            loss = self.loss_function(x, u) + self.eps * tf.reduce_sum(self.head**2)
        grads = tape.gradient(loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)

    def train(self, x, u, niter=10000):
        x = tf.constant(x, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = self.train_op
        loss_op = tf.function(lambda : self.loss_function(x, u))
        loss = []
        min_loss = 1000

        for it in range(niter):
            loss_value = train_op(x, u)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss


class NN(tf.keras.Model):
    """Regular NN."""

    def __init__(self, eps=0.1, name="barebone_nn"):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(50, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
            ]
        )

        self.eps = eps
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        return self.nn.call(x)
    
    def loss_function(self, x, u):
        return tf.reduce_mean((self.call(x) - u) ** 2)

    @tf.function
    def train_op(self, x, u):
        with tf.GradientTape() as tape:
            # L2 regularization
            loss = self.loss_function(x, u) + self.eps * tf.reduce_sum(self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)

    def train(self, x, u, niter=10000):
        x = tf.constant(x, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = self.train_op
        loss_op = tf.function(lambda : self.loss_function(x, u))
        loss = []
        min_loss = 1000

        for it in range(niter):
            loss_value = train_op(x, u)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss


class LA(tf.keras.Model):
    """Model for downstream tasks, with Laplace approximation."""

    def __init__(self, mhnn, flow, dim, noise):
        super().__init__()
        self.body = mhnn.shared_nn
        self.log_prob_fn = flow.log_prob
        self.dim = dim

        init = flow.sample([1])
        self.head = tf.Variable(tf.transpose(init), dtype=tf.float32)

        self.dist = tfp.distributions.Normal(loc=0, scale=noise)

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    def call(self, inputs):
        body = self.body(inputs)
        return tf.matmul(body, self.head[:self.dim]) + self.head[self.dim:]

    def neg_log_posterior(self, inputs, targets):
        log_prior = tf.reduce_sum(self.log_prob_fn(tf.transpose(self.head)))
        log_likelihood = tf.reduce_sum(self.dist.log_prob(self.call(inputs) - targets))
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