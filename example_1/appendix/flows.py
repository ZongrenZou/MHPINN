import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
import numpy as np
import time


tfd = tfp.distributions
tfb = tfp.bijectors


class Coupling(tf.keras.layers.Layer):
    """Coupling layer for RealNVP."""

    def __init__(self, dim, num_masked, hidden_layers, activation=tf.nn.relu):
        super().__init__()
        self.dim = dim
        self.num_masked = num_masked

        self.nn = []
        for unit in hidden_layers:
            self.nn += [
                tf.keras.layers.Dense(
                    unit,
                    activation=activation,
                    kernel_regularizer=tf.keras.regularizers.L2(1.0),
                ),
            ]
        self.nn += [
            tf.keras.layers.Dense(
                2 * (self.dim - self.num_masked),
                kernel_regularizer=tf.keras.regularizers.L2(1.0),
            )
        ]
        self.nn = tf.keras.Sequential(self.nn)
        self.build(input_shape=[None, self.num_masked])

    def call(self, x, output_units):
        shift, log_scale = tf.split(self.nn.call(x), 2, axis=-1)
        log_scale = tf.tanh(log_scale)
        return shift, log_scale


class MADE(tf.keras.layers.Layer):
    def __init__(self, event_shape, hidden_layers, activation):
        super().__init__()

        self.nn = tfb.AutoregressiveNetwork(
            params=2,
            event_shape=event_shape,
            hidden_units=hidden_layers,
            activation=activation,
        )

    def call(self, x):
        shift, log_scale = tf.unstack(self.nn(x), num=2, axis=-1)
        # log_scale = tf.clip_by_value(log_scale, -5, 2)
        log_scale = tf.tanh(log_scale)
        return shift, log_scale


class Flow(tf.keras.Model):
    """Base class for normalizing flows."""

    def __init__(
        self, dim, permutation, hidden_layers, num_bijectors, activation, opt, eps, name
    ):
        super().__init__()
        self.dim = dim
        self.permutation = permutation
        self.hidden_layers = hidden_layers
        self.num_bijectors = num_bijectors
        self.activation = activation
        self.base = tfd.MultivariateNormalDiag(loc=tf.zeros([dim]))
        self._sample = None
        self._log_prob = None
        self.opt = opt
        self.eps = eps
        self._name = name

        self.mu = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
        self.std = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
        self.scale = 0.05

    def losses(self):
        """Overwrites tf.keras.Model.losses."""
        return 0

    def log_prob(self, x):
        log_prob = self._log_prob((x - self.mu) / self.std * self.scale)
        return log_prob + tf.reduce_sum(tf.math.log(self.scale / self.std))

    def sample(self, sample_shape=[]):
        return self.std / self.scale * self._sample(sample_shape) + self.mu

    def train_op(self, x):
        with tf.GradientTape() as tape:
            log_likeli = self.log_prob(x)
            NLL = -tf.reduce_mean(log_likeli) + self.eps * self.losses()
        grads = tape.gradient(NLL, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return NLL

    def restore(self):
        self.load_weights("./checkpoints/" + self.name)

    def train_batch(self, x, batch_size=100, nepoch=1000):
        x = tf.constant(x, tf.float32)
        self.mu.assign(tf.constant(np.mean(x, axis=0), tf.float32))
        self.std.assign(tf.constant(np.std(x, axis=0), tf.float32))

        N = x.shape[0]
        train_op = tf.function(self.train_op)
        log_prob_op = tf.function(self.log_prob)
        min_loss = 2000

        for epoch in range(nepoch):
            idx = np.random.choice(N, N, replace=False)
            t0 = time.time()
            for i in range(N // batch_size):
                batch_id = idx[i * batch_size : (i + 1) * batch_size]
                batch_x = tf.gather(x, batch_id, axis=0)
                _ = train_op(batch_x)
            NLL = -tf.reduce_mean(log_prob_op(x))
            if NLL < min_loss:
                min_loss = NLL
                self.save_weights(
                    filepath="./checkpoints/" + self.name, overwrite=True,
                )
            print(epoch, NLL.numpy())
            print("Elapsed: ", time.time() - t0)


class RealNVP(Flow):
    "RealNVP normalizing flows."

    def __init__(
        self,
        dim,
        num_masked,
        permutation,
        hidden_layers,
        num_bijectors=5,
        activation=tf.nn.relu,
        opt=tf.keras.optimizers.Adam(1e-3),
        eps=0,
        name="realnvp",
    ):
        super().__init__(
            dim, permutation, hidden_layers, num_bijectors, activation, opt, eps, name
        )
        bijectors = []
        self.coupling_layers = []
        for i in range(self.num_bijectors):
            self.coupling_layers += [
                Coupling(
                    dim=self.dim,
                    num_masked=num_masked,
                    hidden_layers=self.hidden_layers,
                    activation=activation,
                ),
            ]
            bijectors += [
                tfb.RealNVP(
                    num_masked=num_masked,
                    shift_and_log_scale_fn=self.coupling_layers[-1],
                ),
                tfb.Permute(self.permutation),
            ]

        self.nvp = tfd.TransformedDistribution(
            distribution=self.base, bijector=tfb.Chain(bijectors),
        )
        _ = self.nvp.sample([1])
        self._sample = tf.function(self.nvp.sample)
        self._log_prob = tf.function(self.nvp.log_prob)

    def losses(self):
        _losses = []
        for layer in self.coupling_layers:
            _losses += layer.nn.losses
        return tf.reduce_sum(_losses)


class MAF(Flow):
    """Masked autoregressive flows."""

    def __init__(
        self,
        dim,
        permutation,
        hidden_layers,
        num_bijectors=5,
        activation=tf.nn.relu,
        opt=tf.keras.optimizers.Adam(1e-3),
        eps=0,
        name="maf",
    ):
        super().__init__(
            dim, permutation, hidden_layers, num_bijectors, activation, opt, eps, name
        )
        bijectors = []
        for i in range(self.num_bijectors):
            bijectors += [
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=MADE(
                        event_shape=[self.dim],
                        hidden_layers=self.hidden_layers,
                        activation=self.activation,
                    ),
                ),
                tfb.Permute(self.permutation),
            ]
        self.maf = tfd.TransformedDistribution(
            distribution=self.base, bijector=tfb.Chain(bijectors),
        )
        _ = self.maf.sample()
        self._sample = tf.function(self.maf.sample)
        self._log_prob = tf.function(self.maf.log_prob)


class IAF(Flow):
    """Inverse autoregressive flows."""

    def __init__(
        self,
        dim,
        permutation,
        hidden_layers,
        num_bijectors=5,
        activation=tf.nn.relu,
        opt=tf.keras.optimizers.Adam(1e-3),
        eps=0,
        name="iaf",
    ):
        super().__init__(
            dim, permutation, hidden_layers, num_bijectors, activation, opt, eps, name
        )
        bijectors = []
        for i in range(self.num_bijectors):
            bijectors += [
                tfb.Invert(
                    tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=MADE(
                            event_shape=[self.dim],
                            hidden_layers=self.hidden_layers,
                            activation=self.activation,
                        ),
                    ),
                ),
                tfb.Permute(self.permutation),
            ]
        self.iaf = tfd.TransformedDistribution(
            distribution=self.base, bijector=tfb.Chain(bijectors),
        )
        _ = self.iaf.sample()
        self._sample = tf.function(self.iaf.sample)
        self._log_prob = tf.function(self.iaf.log_prob)


# class RealNVP(tf.keras.Model):
#     def __init__(
#         self,
#         dim,
#         permutation,
#         hidden_layers,
#         num_bijectors=5,
#         activation=tf.nn.relu,
#         opt=tf.keras.optimizers.Adam(1e-3),
#         name="realnvp",
#     ):
#         super().__init__()

#         bijectors = []
#         shift_and_log_scale_fns = []
#         for i in range(self.num_bijectors):
#             shift_and_log_scale_fns += [
#                 real_nvp_default_template(
#                     hidden_layers=hidden_layers, activation=tf.nn.relu,
#                 )
#             ]
#             bijectors += [
#                 tfb.RealNVP(
#                     fraction_masked=0.5,
#                     shift_and_log_scale_fn=shift_and_log_scale_fns[-1],
#                 ),
#                 tfb.Permute(permutation),
#             ]
#         self.base = tfd.MultivariateNormalDiag(loc=tf.zeros([dim]))
#         self.nvp = tfd.TransformedDistribution(
#             distribution=self.base, bijector=tfb.Chain(bijectors[:-1]),
#         )
#         _ = self.nvp.sample()
#         self._sample = tf.function(self.nvp.sample)
#         self._log_prob = tf.function(self.nvp.log_prob)

#         self._trainable_variables = []
#         for fn in shift_and_log_scale_fns:
#             self._trainable_variables += fn.trainable_variables

#         self.opt = opt
#         self._name = name

#         self.mu = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
#         self.std = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
#         self.scale = 0.05

#     def log_prob(self, x):
#         log_prob = self._log_prob((x - self.mu) / self.std * self.scale)
#         return log_prob + tf.reduce_sum(tf.math.log(self.scale / self.std))

#     def sample(self, sample_shape=[]):
#         return self.std / self.scale * self._sample(sample_shape) + self.mu

#     def train_op(self, x):
#         with tf.GradientTape() as tape:
#             log_likeli = self.log_prob(x)
#             NLL = -tf.reduce_mean(log_likeli)
#         grads = tape.gradient(NLL, self.trainable_variables)
#         self.opt.apply_gradients(zip(grads, self.trainable_variables))
#         return NLL

#     def restore(self):
#         self.load_weights("./checkpoints/" + self.name)

#     def train_batch(self, x, batch_size=100, nepoch=1000):
#         x = tf.constant(x, tf.float32)
#         self.mu.assign(tf.constant(np.mean(x, axis=0), tf.float32))
#         self.std.assign(tf.constant(np.std(x, axis=0), tf.float32))

#         N = x.shape[0]
#         train_op = tf.function(self.train_op)
#         log_prob_op = tf.function(self.log_prob)
#         min_loss = 1000

#         for epoch in range(nepoch):
#             idx = np.random.choice(N, N, replace=False)
#             for i in range(N // batch_size):
#                 batch_id = idx[i * batch_size : (i + 1) * batch_size]
#                 batch_x = tf.gather(x, batch_id, axis=0)
#                 _ = train_op(batch_x)
#             NLL = -tf.reduce_mean(log_prob_op(x))
#             if NLL < min_loss:
#                 min_loss = NLL
#                 self.save_weights(
#                     filepath="./checkpoints/" + self.name, overwrite=True,
#                 )
#             print(epoch, NLL.numpy())


# class MAF(tf.keras.Model):
#     def __init__(
#         self,
#         dim,
#         permutation,
#         hidden_layers,
#         num_bijectors=5,
#         activation=tf.nn.relu,
#         name="maf",
#         opt=tf.keras.optimizers.Adam(1e-3),
#     ):
#         super().__init__()
#         bijectors = []
#         self.num_bijectors = num_bijectors
#         for i in range(num_bijectors):
#             bijectors += [
#                 tfb.MaskedAutoregressiveFlow(
#                     shift_and_log_scale_fn=MADE(
#                         event_shape=[dim],
#                         hidden_layers=hidden_layers,
#                         activation=activation,
#                     )
#                 ),
#                 tfb.Permute(permutation),
#             ]

#         self.base = tfd.MultivariateNormalDiag(loc=tf.zeros([dim]))
#         self.maf = tfd.TransformedDistribution(
#             distribution=self.base, bijector=tfb.Chain(bijectors),
#         )
#         _ = self.maf.sample()
#         self._sample = tf.function(self.maf.sample)
#         self._log_prob = tf.function(self.maf.log_prob)

#         # for data normalization
#         self.mu = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
#         self.std = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
#         self.scale = 0.05
#         # self.mu = 0
#         # self.std = 1
#         # self.scale = 1/4

#         self._name = name
#         self.opt = opt

#     def log_prob(self, x):
#         log_prob = self._log_prob((x - self.mu) / self.std * self.scale)
#         return log_prob + tf.reduce_sum(tf.math.log(self.scale / self.std))

#     def sample(self, sample_shape=[]):
#         return self.std / self.scale * self._sample(sample_shape) + self.mu

#     def train_op(self, x):
#         with tf.GradientTape() as tape:
#             log_likeli = self.log_prob(x)
#             NLL = -tf.reduce_mean(log_likeli)
#         grads = tape.gradient(NLL, self.trainable_variables)
#         self.opt.apply_gradients(zip(grads, self.trainable_variables))
#         return NLL

#     def train_batch(self, x, batch_size=100, nepoch=1000):
#         x = tf.constant(x, tf.float32)
#         self.mu.assign(tf.constant(np.mean(x, axis=0), tf.float32))
#         self.std.assign(tf.constant(np.std(x, axis=0), tf.float32))
#         N = x.shape[0]
#         train_op = tf.function(self.train_op)
#         log_prob_op = tf.function(self.log_prob)

#         min_loss = 1000

#         for epoch in range(nepoch):
#             idx = np.random.choice(N, N, replace=False)
#             for i in range(N // batch_size):
#                 batch_id = idx[i * batch_size : (i + 1) * batch_size]
#                 batch_x = tf.gather(x, batch_id, axis=0)
#                 _ = train_op(batch_x)
#             NLL = -tf.reduce_mean(log_prob_op(x))
#             if NLL < min_loss:
#                 min_loss = NLL
#                 self.save_weights(
#                     filepath="./checkpoints/" + self.name, overwrite=True,
#                 )
#             print(epoch, NLL.numpy())

#     def restore(self):
#         self.load_weights("./checkpoints/" + self.name)
