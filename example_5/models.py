import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
import time


# tfd = tfp.distributions
# tfb = tfp.bijectors


def jvp(y, x, v):
    # For more information, see https://github.com/renmengye/tensorflow-forward-ad/issues/2
    u = tf.ones_like(y) # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class MHPINN(tf.keras.Model):
    """Multi-head PINN."""

    def __init__(self, num_tasks=1000, dim=100, name="meta"):
        super().__init__()
        self.shared_nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
                tf.keras.layers.Dense(dim, activation=tf.tanh),
            ]
        )
        self.dim = dim
        self.N = num_tasks
        self.heads = tf.Variable(0.05 * tf.random.normal(shape=[dim+1, self.N]), dtype=tf.float32)
        # self.xi = tf.Variable(0.05 * tf.random.normal(shape=[self.N, dim+1]), dtype=tf.float32)
        self.shared_nn.build(input_shape=[None, 2])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, y, heads):
        shared = self.shared_nn.call(tf.concat([x, y], axis=-1))
        out = tf.matmul(shared, heads[:self.dim, :]) + heads[self.dim, :]
        out = x * (2*np.pi-x) * y * (2*np.pi-y) * out
        return out / np.pi ** 4

    @tf.function
    def pde(self, x, y, heads):
        u = self.call(x, y, heads)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return u - u_xx - u_yy

    def loss_function(self, x, y, f, heads):
        u = self.call(x, y, heads)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return tf.reduce_mean((u - u_xx - u_yy - f) ** 2)

    @tf.function
    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x, y, f, self.heads)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x_train, y_train, f_train, niter=10000, ftol=5e-5):
        x_train = tf.constant(x_train, tf.float32)
        y_train = tf.constant(y_train, tf.float32)
        f_train = tf.constant(f_train, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x_train, y_train, f_train)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1], ", time: ", time.time() - t0, flush=True)
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


class PINN(tf.keras.Model):

    def __init__(self, units=100, name="pinn", eps=0.0):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(units, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(units, activation=tf.tanh, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
                tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(1.0)),
            ]
        )
        self.nn.build(input_shape=[None, 2])
        self._name = name
        self.eps = eps
        self.opt = tf.keras.optimizers.Adam(1e-3)

    def call(self, x, y):
        out = self.nn(tf.concat([x, y], axis=-1))
        out = x * (2*np.pi-x) * y * (2*np.pi-y) * out 
        return out / np.pi ** 4

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

    def loss_function(self, x, y, f):
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
        return tf.reduce_mean((u - u_xx - u_yy - f)**2)

    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x, y, f) + self.eps * tf.reduce_sum(self.losses)
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
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class Downstream(tf.keras.Model):
    """For downstream tasks."""

    def __init__(self, mhnn, flow, dim, eps=0.1, name="downstream"):
        super().__init__()
        self.body = mhnn
        self.log_prob_fn = flow.log_prob
        self.sample_fn = flow.sample
        self.dim = dim

        init = self.sample_fn([1])

        self.head = tf.Variable(tf.transpose(init), dtype=tf.float32)
        self.eps = eps
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, y):
        shared = self.body.shared_nn(tf.concat([x, y], axis=-1))
        out = tf.matmul(shared, self.head[:self.dim, :]) + self.head[self.dim, :]
        out = x * (2*np.pi-x) * y * (2*np.pi-y) * out
        return out / np.pi ** 4

    @tf.function
    def pde(self, x, y):
        u = self.call(x, y)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return u - u_xx - u_yy

    def loss_function(self, x, y, f):
        u = self.call(x, y)
        v = tf.ones_like(x)
        u_x = jvp(u, x, v)[0]
        u_xx = jvp(u_x, x, v)[0]
        u_y = jvp(u, y, v)[0]
        u_yy = jvp(u_y, y, v)[0]
        return tf.reduce_mean((u - u_xx - u_yy - f) ** 2)

    @tf.function
    def train_op(self, x, y, f):
        with tf.GradientTape() as tape:
            regularization = - tf.math.reduce_sum(
                self.log_prob_fn(tf.transpose(self.head))
            )
            loss = self.loss_function(x, y, f) + self.eps * regularization
        grads = tape.gradient(loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return loss

    def train(self, x_train, y_train, f_train, niter=10000):
        x_train = tf.constant(x_train, tf.float32)
        y_train = tf.constant(y_train, tf.float32)
        f_train = tf.constant(f_train, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x_train, y_train, f_train)
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

    @tf.function
    def train_batch_op(self, x, y, f, batch_idx):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x, y, f, tf.gather(self.heads, batch_idx, axis=-1))
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train_batch(self, x_train, y_train, f_train, nepoch=1000, batch_size=100, ftol=5e-5):
        x_train = tf.constant(x_train, tf.float32)
        y_train = tf.constant(y_train, tf.float32)
        f_train = tf.constant(f_train, tf.float32)
        num_tasks = f_train.shape[1]

        train_op = self.train_batch_op
        loss_op = tf.function(self.loss_function)
        loss = []
        min_loss = 1000

        t0 = time.time()
        for epoch in range(nepoch):
            idx = np.random.choice(num_tasks, num_tasks, replace=False)
            idx = tf.constant(idx, tf.int32)
            for i in range(num_tasks//batch_size):
                batch_idx = idx[i*batch_size:(i+1)*batch_size]
                f_train_batch = tf.gather(f_train, batch_idx, axis=-1)
                _ = train_op(
                    x_train, y_train, f_train_batch, batch_idx
                )
            loss_value = loss_op(x_train, y_train, f_train, self.heads).numpy()
            loss += [loss_value]
            print("Epoch: ", epoch, ", loss: ", loss_value, ", time: ", time.time()-t0)
            if loss_value < min_loss:
                min_loss = loss_value
                self.save_weights(
                    filepath="./checkpoints/"+self.name,
                    overwrite=True,
                )
            t0 = time.time()
        return loss
