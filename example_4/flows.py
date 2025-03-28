import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
import numpy as np
import time


from tensorflow_probability.python.internal import tensorshape_util


tfd = tfp.distributions
tfb = tfp.bijectors


def real_nvp_default_template(
    hidden_layers, activation=tf.nn.relu, name=None,
):

    with tf.name_scope(name or "real_nvp_default_template"):

        def _fn(x, output_units, **condition_kwargs):
            if tensorshape_util.rank(x.shape) == 1:
                x = x[tf.newaxis, ...]
                reshape_output = lambda x: x[0]
            else:
                reshape_output = lambda x: x
            for units in hidden_layers:
                x = tf1.layers.dense(inputs=x, units=units, activation=activation,)
            x = tf1.layers.dense(inputs=x, units=2 * output_units, activation=None,)
            shift, log_scale = tf.split(x, 2, axis=-1)
            log_scale = tf.clip_by_value(log_scale, -2, 2) # to avoid gradient explosion
            # log_scale = tf.math.tanh(log_scale)
            return reshape_output(shift), reshape_output(log_scale)

        return tf1.make_template("real_nvp_default_template", _fn)


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
        log_scale = tf.clip_by_value(log_scale, -2, 2)
        return shift, log_scale


class RealNVP(tf.keras.Model):
    def __init__(
        self, dim, permutation, hidden_layers, num_bijectors=5, activation=tf.nn.relu, opt=tf.keras.optimizers.Adam(1e-3),
    ):
        super().__init__()
        bijectors = []
        shift_and_log_scale_fns = []
        self.num_bijectors = num_bijectors
        for i in range(num_bijectors):
            shift_and_log_scale_fns += [
                real_nvp_default_template(
                    hidden_layers=hidden_layers, activation=tf.nn.relu,
                )
            ]
            bijectors += [
                tfb.RealNVP(
                    fraction_masked=0.5,
                    shift_and_log_scale_fn=shift_and_log_scale_fns[-1],
                ),
                tfb.Permute(permutation),
            ]
        self.base = tfd.MultivariateNormalDiag(loc=tf.zeros([dim]))
        self.nvp = tfd.TransformedDistribution(
            distribution=self.base, bijector=tfb.Chain(bijectors[:-1]),
        )
        _ = self.nvp.sample()
        
        self._trainable_variables = []
        for fn in shift_and_log_scale_fns:
            self._trainable_variables += fn.trainable_variables

        self.opt = opt

    def sample(self, sample_size):
        return self.nvp.sample(sample_size)

    def log_prob(self, x):
        return self.nvp.log_prob(x)

    def train_op(self, x):
        with tf.GradientTape() as tape:
            log_likeli = self.log_prob(x)
            NLL = -tf.reduce_mean(log_likeli)
        grads = tape.gradient(NLL, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return NLL

    def restore(self):
        self.load_weights("./checkpoints/nf")

    def train_batch(self, x, batch_size=100, nepoch=1000):
        x = tf.constant(x, tf.float32)
        N = x.shape[0]
        train_op = tf.function(self.train_op)
        log_prob_op = tf.function(self.log_prob)
        min_loss = 1000

        for epoch in range(nepoch):
            idx = np.random.choice(N, N, replace=False)
            for i in range(N // batch_size):
                batch_id = idx[i * batch_size : (i + 1) * batch_size]
                batch_x = tf.gather(x, batch_id, axis=0)
                _ = train_op(batch_x)
            NLL = -tf.reduce_mean(log_prob_op(x))
            if NLL < min_loss:
                min_loss = NLL
                self.save_weights(
                    filepath="./checkpoints/nf", overwrite=True,
                )
            print(epoch, NLL.numpy())
            

class MAF(tf.keras.Model):
    def __init__(self, dim, permutation, hidden_layers, num_bijectors=5, activation=tf.nn.relu, opt=tf.keras.optimizers.Adam(1e-3)):
        super().__init__()
        bijectors = []
        self.num_bijectors = num_bijectors
        for i in range(num_bijectors):
            bijectors += [
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=MADE(
                        event_shape=[dim], hidden_layers=hidden_layers, activation=activation,
                    )
                ),
                tfb.Permute(permutation),
            ]
    
        self.base = tfd.MultivariateNormalDiag(loc=tf.zeros([dim]))
        self.maf = tfd.TransformedDistribution(
            distribution=self.base,
            bijector=tfb.Chain(bijectors),
        )
        _ = self.maf.sample()
        self._sample = tf.function(self.maf.sample)
        self._log_prob = tf.function(self.maf.log_prob)
        
        # for data normalization
        self.mu = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
        self.std = tf.Variable(tf.zeros(shape=[dim]), trainable=False)
        # self.mu = None
        # self.std = None
        self.scale = 1/4

        self.opt = opt
    
    def log_prob(self, x):
        log_prob = self._log_prob((x - self.mu) / self.std * self.scale)
        return log_prob + tf.reduce_sum(tf.math.log(self.scale/self.std))

    def sample(self, sample_shape=[]):
        return self.std / self.scale * self._sample(sample_shape) + self.mu

    def train_op(self, x):
        with tf.GradientTape() as tape:
            log_likeli = self.log_prob(x)
            NLL = - tf.reduce_mean(log_likeli)
        grads = tape.gradient(NLL, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return NLL
    
    def train_batch(self, x, batch_size=100, nepoch=1000):
        x = tf.constant(x, tf.float32)
        self.mu = tf.Variable(
            np.mean(x, axis=0), dtype=tf.float32, trainable=False,
        )
        self.std = tf.Variable(
            np.std(x, axis=0), dtype=tf.float32, trainable=False,
        )
        # self.mu = tf.constant(np.mean(x, axis=0), tf.float32)
        # self.std = tf.constant(np.std(x, axis=0), tf.float32)
        N = x.shape[0]
        train_op = tf.function(self.train_op)
        log_prob_op = tf.function(self.log_prob)

        min_loss = 1000
        
        for epoch in range(nepoch):
            idx = np.random.choice(N, N, replace=False)
            for i in range(N//batch_size):
                batch_id = idx[i*batch_size:(i+1)*batch_size]
                batch_x = tf.gather(x, batch_id, axis=0)
                _ = train_op(batch_x)
            NLL = -tf.reduce_mean(log_prob_op(x))
            if NLL < min_loss:
                min_loss = NLL
                self.save_weights(
                    filepath="./checkpoints/maf", overwrite=True,
                )
            print(epoch, NLL.numpy())
    
    def restore(self):
        self.load_weights("./checkpoints/maf")