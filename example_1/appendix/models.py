import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time


tfd = tfp.distributions
tfb = tfp.bijectors


class Meta(tf.keras.Model):

    def __init__(self, num_tasks=1000, dim=100, name="meta"):
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
        # self.heads = tf.Variable(tf.random.normal(shape=[dim+1, self.N]), dtype=tf.float32)
        self.shared_nn.build(input_shape=[None, 1])
        self._name = name

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs, heads):
        """
        Computes a forward pass.

            Args:
                inputs: The inputs to the neural networks, with shape [batch_size, input_dim]
                heads: The heads of the neural networks, which share the same body, with 
                    shape [latent_dim+1, num_heads], where "+1" is for bias.
        
            Returns:
                y: The outputs to the neural networks, with shape [batch_size, num_heads]
        """
        shared = self.shared_nn.call(inputs) # with shape [batch_size, latent_dim]
        y = tf.matmul(shared, heads[:self.dim, :]) + heads[self.dim:, :]
        return y

    def loss_function(self, x1, y1, x2, y2):
        heads1, heads2 = tf.split(self.heads, 2, axis=-1)
        out1 = self.call(x1, heads1)
        out2 = self.call(x2, heads2)
        return tf.reduce_mean((out1 - y1) ** 2) + tf.reduce_mean((out2 - y2) ** 2)

    @tf.function
    def train_op(self, x1, y1, x2, y2):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x1, y1, x2, y2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x1, y1, x2, y2, niter=10000, ftol=5e-5):
        x1 = tf.constant(x1, tf.float32)
        x2 = tf.constant(x2, tf.float32)
        y1 = tf.constant(y1, tf.float32)
        y2 = tf.constant(y2, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x1, y1, x2, y2)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1], ", time: ", time.time() - t0)
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
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


class Model(tf.keras.Model):

    def __init__(self, body, dim, eps=0.0, scale=1.0):
        super().__init__()
        self.body = body
        self.dim = dim
        self.head = tf.Variable(scale * tf.random.normal(shape=[dim+1, 1]), dtype=tf.float32)
        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    def call(self, inputs):
        body = self.body.shared_nn(inputs)
        return tf.matmul(body, self.head[:self.dim]) + self.head[self.dim:]
    
    @tf.function
    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((targets - self.call(inputs)) ** 2)
            regularization = tf.math.reduce_sum(self.head ** 2)
            total_loss = loss + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return loss

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
                if loss_value < min_loss:
                    min_loss = loss_value
                    self.save_weights(
                        filepath="./checkpoints/model",
                        overwrite=True,
                    )

        return loss

    def restore(self):
        self.load_weights("./checkpoints/model")


class Model2(tf.keras.Model):
    """Model for downstream tasks, with regularization."""

    def __init__(self, body, flow, dim, eps=0.0):
        super().__init__()
        self.body = body
        self.flow = flow
        self.log_prob_fn = tf.function(flow.log_prob)
        self.dim = dim
        self.head = tf.Variable(tf.transpose(flow.sample([1])), dtype=tf.float32)
        # self.xi = tf.Variable(flow._sample([1]), dtype=tf.float32)

        self.eps = eps

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    def call(self, inputs):
        body = self.body.shared_nn(inputs)
        return tf.matmul(body, self.head[:self.dim]) + self.head[self.dim:]
    
    @tf.function
    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((targets - self.call(inputs)) ** 2)
            regularization = - tf.math.reduce_sum(self.log_prob_fn(tf.transpose(self.head)))
            total_loss = loss + self.eps * regularization
        grads = tape.gradient(total_loss, [self.head])
        self.opt.apply_gradients(zip(grads, [self.head]))
        return loss

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
                        filepath="./checkpoints/model2",
                        overwrite=True,
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
        body = self.body.shared_nn(inputs)
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


# class Model3(tf.keras.Model):
#     """Model for downstream tasks, with Laplace approximation."""

#     def __init__(self, body, flow, dim, noise, scale=1.0):
#         super().__init__()
#         self.body = body
#         self.log_prob_fn = tf.function(flow.log_prob)
#         self.sample_fn = tf.function(flow.sample)
#         self.dim = dim
#         init = self.sample_fn([1])
#         self.head = tf.Variable(tf.transpose(init))
#         self.dist = tfd.Normal(loc=0, scale=noise)

#         self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
#     def call(self, inputs):
#         body = self.body.shared_nn(inputs)
#         return tf.matmul(body, self.head[:self.dim]) + self.head[self.dim:]

#     def log_posterior(self, inputs, targets):
#         log_prior = tf.reduce_sum(self.log_prob_fn(self.head))
#         log_likelihood = tf.reduce_sum(self.dist.log_prob(self.call(inputs) - targets))
#         return log_prior + log_likelihood

#     @tf.function
#     def train_op(self, inputs, targets):
#         with tf.GradientTape() as tape:
#             log_posterior = self.log_posterior(inputs, targets)
#         grads = tape.gradient(log_posterior, [self.head])
#         self.opt.apply_gradients(zip(grads, [self.head]))
#         return log_posterior

#     def train(self, inputs, targets, niter=10000):
#         inputs_train = tf.constant(inputs, tf.float32)
#         targets_train = tf.constant(targets, tf.float32)
#         train_op = self.train_op

#         min_loss = 1000
#         loss = []

#         for it in range(niter):
#             loss_value = train_op(inputs_train, targets_train)
#             loss += [loss_value.numpy()]
#             if it % 1000 == 0:
#                 print(it, loss[-1])
#                 if loss_value < min_loss and it > niter//2:
#                     min_loss = loss_value
#                     self.save_weights(
#                         filepath="./checkpoints/model3",
#                         overwrite=True,
#                     )

#         return loss

#     def restore(self):
#         self.load_weights("./checkpoints/model3")

#     def posterior_sample(self, sample_size, inputs, targets):
#         with tf.GradientTape() as g_xx:
#             with tf.GradientTape() as g_x:
#                 L = self.log_posterior(inputs, targets)
#             J = g_x.gradient(L, self.head)
#         H = g_xx.gradient(J, self.head)


class Meta2(tf.keras.Model):

    def __init__(self, num_tasks=1000, dim=100, eps=0.0, scale=1.0):
        super().__init__()
        self.shared_nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(1)
            ]
        )
        self.dim = dim
        self.eps = eps
        self.N = num_tasks
        self.ws = tf.Variable(scale * tf.random.normal(shape=[self.N, 1, dim]), dtype=tf.float32)
        self.bs = tf.Variable(tf.zeros(shape=[self.N, 1, dim]), dtype=tf.float32)
        self.shared_nn.build(input_shape=[None, None, dim])

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, ws, bs):
        """
        Computes a forward pass.

            Args:
                x: The inputs to the neural networks, with shape [batch_size, input_dim]
                ws: The weights of the first layer, with shape [num_heads, input_dim, latent_dim].
                bs: The biases of the first layer, with shape [num_heads, 1, latent_dim].
        
            Returns:
                y: The outputs to the neural networks, with shape [batch_size, num_heads, 1]
        """
        output_of_first_layer = tf.tanh(
            tf.einsum("Bi,Nij->BNj", x, ws) + tf.transpose(bs, [1, 0, 2])
        ) # shape of [batch_size, num_heads, latent_dim]
        out = self.shared_nn(output_of_first_layer)
        return out[..., 0]

    def loss_function(self, x1, y1, x2, y2):
        ws1, ws2 = tf.split(self.ws, 2, axis=0)
        bs1, bs2 = tf.split(self.bs, 2, axis=0)

        out1 = self.call(x1, ws1, bs1)
        out2 = self.call(x2, ws2, bs2)
        return tf.reduce_mean((out1 - y1) ** 2) + tf.reduce_mean((out2 - y2) ** 2)

    @tf.function
    def train_op(self, x1, y1, x2, y2):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x1, y1, x2, y2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x1, y1, x2, y2, niter=10000, ftol=5e-5):
        x1 = tf.constant(x1, tf.float32)
        x2 = tf.constant(x2, tf.float32)
        y1 = tf.constant(y1, tf.float32)
        y2 = tf.constant(y2, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x1, y1, x2, y2)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1], ", time: ", time.time() - t0)
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/meta2",
                        overwrite=True,
                    )
                if loss[-1] < ftol:
                    break
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/meta2")


class Meta3(tf.keras.Model):

    def __init__(self, num_tasks=1000, dim=100, eps=0.0, scale=1.0):
        super().__init__()
        self.shared_nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(1)
            ]
        )
        self.dim = dim
        self.eps = eps
        self.N = num_tasks
        self.ws = tf.Variable(scale * tf.random.normal(shape=[self.N, 1, dim]), dtype=tf.float32)
        self.bs = tf.Variable(tf.zeros(shape=[1, 1, dim]), dtype=tf.float32)
        # self.bs = tf.Variable(tf.zeros(shape=[self.N, 1, dim]), dtype=tf.float32)
        self.shared_nn.build(input_shape=[None, None, dim])

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x, ws):
        """
        Computes a forward pass.

            Args:
                x: The inputs to the neural networks, with shape [batch_size, input_dim]
                ws: The weights of the first layer, with shape [num_heads, input_dim, latent_dim].
                bs: The biases of the first layer, with shape [num_heads, 1, latent_dim].
        
            Returns:
                y: The outputs to the neural networks, with shape [batch_size, num_heads, 1]
        """
        output_of_first_layer = tf.tanh(
            tf.einsum("Bi,Nij->BNj", x, ws) + self.bs
        ) # shape of [batch_size, num_heads, latent_dim]
        out = self.shared_nn(output_of_first_layer)
        return out[..., 0]

    def loss_function(self, x1, y1, x2, y2):
        ws1, ws2 = tf.split(self.ws, 2, axis=0)

        out1 = self.call(x1, ws1)
        out2 = self.call(x2, ws2)
        return tf.reduce_mean((out1 - y1) ** 2) + tf.reduce_mean((out2 - y2) ** 2)

    @tf.function
    def train_op(self, x1, y1, x2, y2):
        with tf.GradientTape() as tape:
            loss = self.loss_function(x1, y1, x2, y2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x1, y1, x2, y2, niter=10000, ftol=5e-5):
        x1 = tf.constant(x1, tf.float32)
        x2 = tf.constant(x2, tf.float32)
        y1 = tf.constant(y1, tf.float32)
        y2 = tf.constant(y2, tf.float32)

        train_op = self.train_op
        loss = []
        min_loss = 1000

        t0 = time.time()
        for it in range(niter):
            loss_value = train_op(x1, y1, x2, y2)
            loss += [loss_value.numpy()]
            if it % 1000 == 0:
                print(it, loss[-1], ", time: ", time.time() - t0)
                if loss[-1] < min_loss:
                    min_loss = loss[-1]
                    self.save_weights(
                        filepath="./checkpoints/meta3",
                        overwrite=True,
                    )
                if loss[-1] < ftol:
                    break
                t0 = time.time()
        return loss

    def restore(self):
        self.load_weights("./checkpoints/meta3")
