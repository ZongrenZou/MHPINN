"""Generate data for GP."""
import numpy as np
import scipy.io as sio


def generate_data_1(l=0.1, M=256, N=10000):
    """
    Generate samples of GP with squared exponential kernel function,
    using Cholesky decomposition.

        Args:
            l (scalar): The length scale for the kernel function.
            M (integer): The number of uniformly distributed sampling 
                points, for samples of GP.
            N (integer): The number of samples.
        Returns:
            x (array): The sampling points, with shape [M, N].
            f (array): The values of GP at the sampling points, with shape
                with shape [M, N].
    """
    x = np.linspace(0, 1, M + 1).reshape([-1, 1])
    C = np.exp(-((x - x.T) ** 2) / 2 / l ** 2) + 1e-10 * np.eye(M + 1)
    L = np.linalg.cholesky(C)
    xi = np.random.normal(size=[M + 1, N])
    f = np.matmul(L, xi)
    return x, f


def generate_data_2(l=0.1, M=256, N=10000, K=5):
    """
    Generate samples of GP with squared exponential kernel function,
    using truncated Karhunen-Loeve (KL) expansion.

        Args:
            l (scalar): The length scale for the kernel function.
            M (integer): The number of uniformly distributed sampling 
                points, for samples of GP.
            N (integer): The number of samples.
            K (integer): The number of truncated KL expansion terms.
        Returns:
            x (array): The sampling points, with shape [M, N].
            f (array): The values of GP at the sampling points, with shape
                with shape [M, N].
    """
    x = np.linspace(0, 1, M + 1).reshape([-1, 1])
    C = np.exp(-((x - x.T) ** 2) / 2 / l ** 2) + 1e-10 * np.eye(M + 1)
    w, v = np.linalg.eig(C)
    w, v = np.real(w), np.real(v)
    ind = np.argsort(-w)
    w = w[ind]
    v = v[:, ind]
    xi = np.random.normal(size=[M + 1, N])
    f = np.zeros([M + 1, N])
    for i in range(K):
        f += np.sqrt(w[i]) * np.matmul(v[:, i : i + 1], xi[i : i + 1, :])
    return x, f


class DataSet:

    def __init__(self, N, batch_size):
        self.batch_size = batch_size
        self.N = N
        file_name = "./data/data_10.mat"
        print("loading data from " + file_name)
        data = sio.loadmat(file_name)
        self.t = data["t"]
        self.f = data["f"]
        self.sols = data["sols"][:, 0, :]
        self.f_train = self.f[::4, :N]
        self.t_train = self.t[::4, :]
    
    def minibatch(self):
        idx_batch = np.random.choice(self.N, self.batch_size, replace=False)
        f_batch = self.f_train[:, idx_batch]
        t_batch = self.t_train
        return t_batch, f_batch
