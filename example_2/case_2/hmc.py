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


class PI_Bayesian:
    def __init__(
        self, t_u, u, t_f, f, pde_fn, meta, flow, noise_u=0.05, noise_f=0.05,
    ):
        self.t_u = t_u
        self.u = u
        self.t_f = t_f
        self.f = f
        self.pde_fn = pde_fn
        self.meta = meta
        self.flow = flow
        self.noise_u = noise_u
        self.noise_f = noise_f

        self.initial_values = self.flow.sample([1])

    def build_posterior(self):
        u = tf.constant(self.u, dtype=tf.float32)
        f = tf.constant(self.f, dtype=tf.float32)

        def _fn(*variables):
            """
            log posterior function, which takes neural network's parameters input, and outputs (probably unnormalized) density probability
            """
            # split the input list into variables for neural networks, and additional variables
            xi = variables[0]
            head = xi[:, :51]
            log_k = xi[:, 51:]
            print(xi.shape, head.shape, log_k.shape)

            t_u = tf.constant(self.t_u, dtype=tf.float32)
            t_f = tf.constant(self.t_f, dtype=tf.float32)

            u_pred = self.meta.call(t_u, tf.transpose(head))
            print(u_pred.shape, self.meta.call(t_f, tf.transpose(head)).shape)
            f_pred = self.pde_fn(t_f, self.meta.call(t_f, tf.transpose(head)), log_k)
            print(f_pred.shape)

            log_prior = tf.reduce_sum(self.flow.log_prob(xi))
            likelihood_u = tfd.Normal(loc=u, scale=self.noise_u)
            likelihood_f = tfd.Normal(loc=f, scale=self.noise_f)
            log_likeli = tf.reduce_sum(likelihood_u.log_prob(u_pred)) + tf.reduce_sum(likelihood_f.log_prob(f_pred))

            return log_prior + log_likeli

        return _fn


class AdaptiveHMC:
    def __init__(
        self,
        target_log_prob_fn,
        init_state,
        num_results=1000,
        num_burnin=1000,
        num_leapfrog_steps=30,
        step_size=0.1,
    ):
        self.target_log_prob_fn = target_log_prob_fn
        self.init_state = init_state
        self.kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
            ),
            num_adaptation_steps=int(0.8 * num_burnin),
            target_accept_prob=0.75,
        )
        self.num_results = num_results
        self.num_burnin = num_burnin

    @tf.function
    def run_chain(self):
        samples, results = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            num_burnin_steps=self.num_burnin,
            current_state=self.init_state,
            kernel=self.kernel,
        )
        return samples, results
