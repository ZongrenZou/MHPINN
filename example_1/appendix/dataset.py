import tensorflow as tf
import numpy as np


# def generate_data(N, M1=32, M2=64):
#     w1 = np.pi + 1 * np.pi * np.random.uniform(size=[N//2, 1])
#     w2 = 4 * np.pi + 1 * np.pi * np.random.uniform(size=[N//2, 1])
#     A = 1 + 2 * np.random.uniform(size=[N, 1])
#     k = 2 * (np.random.uniform(size=[N, 1]) > 0.5).astype(np.float32) - 1

#     x1 = np.linspace(-1, 1, M1)
#     x2 = np.linspace(-1, 1, M2)
#     xx1 = np.tile(x1[None, ...], [N//2, 1])
#     xx2 = np.tile(x2[None, ...], [N//2, 1])

#     yy1 = A[:N//2, :] * np.cos(w1*xx1) + 2 * k[:N//2, :] * xx1
#     yy2 = A[N//2:, :] * np.cos(w2*xx2) + 2 * k[N//2:, :] * xx2

#     return xx1, yy1, xx2, yy2, w1, w2, A, k


def generate_data(N, M1=32, M2=64):
    w = 2 * np.pi + 2 * np.pi * np.random.uniform(size=[N, 1])
    A = 1 + 2 * np.random.uniform(size=[N, 1])
    k = 2 * (np.random.uniform(size=[N, 1]) > 0.5).astype(np.float32) - 1

    x1 = np.linspace(-1, 1, M1)
    x2 = np.linspace(-1, 1, M2)
    w1 = w[w<=3*np.pi].reshape([-1, 1])
    w2 = w[w>3*np.pi].reshape([-1, 1])
    xx1 = np.tile(x1[None, ...], [w1.shape[0], 1])
    xx2 = np.tile(x2[None, ...], [w2.shape[0], 1])

    yy1 = A[:w1.shape[0], :] * np.cos(w1*xx1) + 2 * k[:w1.shape[0], :] * xx1
    yy2 = A[w1.shape[0]:, :] * np.cos(w2*xx2) + 2 * k[w1.shape[0]:, :] * xx2

    return xx1, yy1, xx2, yy2, w1, w2, A, k


def generate_data(N, M1=32, M2=64, upper=4*np.pi):
    w = 2 * np.pi + (upper - 2*np.pi) * np.random.uniform(size=[N, 1])
    A = 1 + 2 * np.random.uniform(size=[N, 1])
    k = 2 * (np.random.uniform(size=[N, 1]) > 0.5).astype(np.float32) - 1

    x1 = np.linspace(-1, 1, M1)
    x2 = np.linspace(-1, 1, M2)
    w1 = w[w<=3*np.pi].reshape([-1, 1])
    w2 = w[w>3*np.pi].reshape([-1, 1])
    xx1 = np.tile(x1[None, ...], [w1.shape[0], 1])
    xx2 = np.tile(x2[None, ...], [w2.shape[0], 1])

    yy1 = A[:w1.shape[0], :] * np.cos(w1*xx1) + 2 * k[:w1.shape[0], :] * xx1
    yy2 = A[w1.shape[0]:, :] * np.cos(w2*xx2) + 2 * k[w1.shape[0]:, :] * xx2

    return xx1, yy1, xx2, yy2, w1, w2, A, k


# def generate_data_1(N, M):
#     w = 3*np.pi + np.pi * np.random.uniform(size=[N, 1])
#     A = 1 + np.random.uniform(size=[N, 1])
#     k = 2 * (np.random.uniform(size=[N, 1]) > 0.5).astype(np.float32) - 1
    
#     x = -1 + 2 * np.random.uniform(size=[M])
#     xx = np.tile(x[None, ...], [N, 1])
    
#     yy = A * np.cos(w*xx) + 1.5 * k * xx
    
#     x_test = np.linspace(-1, 1, 1000)
#     xx_test = np.tile(x_test[None, ...], [N, 1])
#     yy_test = A * np.cos(w*xx_test) + 1.5 * k * xx_test
    
#     return xx, yy, w, A, k, xx_test, yy_test


# def generate_data_3(N, M):
#     w = 3*np.pi + np.pi * np.random.uniform(size=[N, 1])
#     A = 1 + np.random.uniform(size=[N, 1])
#     k = 2 * (np.random.uniform(size=[N, 1]) > 0.5).astype(np.float32) - 1
    
#     xx = np.zeros([N, M])
#     for i in range(N):
#         xx[i, :] = -1 + 2 * np.random.uniform(size=[M])
    
#     yy = A * np.cos(w*xx) + 1.5 * k * xx
    
#     x_test = np.linspace(-1, 1, 1000)
#     xx_test = np.tile(x_test[None, ...], [N, 1])
#     yy_test = A * np.cos(w*xx_test) + 1.5 * k * xx_test
    
#     return xx, yy, w, A, k, xx_test, yy_test


# def generate_data_2(N):
#     """Half-half."""
#     w = 3*np.pi + np.pi * np.random.uniform(size=[N, 1])
#     A = 1 + np.random.uniform(size=[N, 1])
#     k = 2 * (np.random.uniform(size=[N, 1]) > 0.5).astype(np.float32) - 1
    
#     # x = np.linspace(-1, 1, 50)
#     # xx = np.tile(x[None, ...], [N, 1])
#     xL = np.linspace(-1, 0.5, 40).reshape([1, -1])
#     xL = np.tile(xL, [N//2, 1])
#     xR = np.linspace(-0.5, 1, 40).reshape([1, -1])
#     xR = np.tile(xR, [N//2, 1])
#     xx = np.concatenate([xL, xR], axis=0)

#     yy = A * np.cos(w*xx) + 1.5 * k * xx
    
#     return xx, yy, w, A, k
