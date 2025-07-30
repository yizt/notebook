import numpy as np
import numpy.matlib as npm
from loguru import logger
from scipy.linalg import eigh

def quat_normalize(q):
    """四元数归一化"""
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    return q / norm

def weighted_quat_mean(quaternions, weights, eps=1e-8):
    """加权四元数均值（带数值稳定性处理）"""
    if len(quaternions) == 0 or len(weights) != len(quaternions):
        raise ValueError("quaternions and weights must have the same non-zero length")
    # 处理无效权重
    weights = np.nan_to_num(weights)
    weights = np.clip(weights, 0, None)  # 确保权重非负
    weights += eps  # 避免全零权重
    weights /= np.sum(weights)

    Q = np.array(quaternions)
    # 处理符号歧义：使第一个四元数为正
    signs = np.sign(Q[:, 0:1])  # 计算每个四元数实部的符号, +1/-1/0
    signs[signs == 0] = 1  # 如果实部为零，则将符号设置为 +1。这一步是为了避免因实部为零而导致的不确定性。
    Q *= signs  # 将每个四元数乘以其对应的符号, 确保所有四元数实部符号一致（均为正）。

    W = np.diag(weights)
    # logger.debug(f"{W=}")
    C = Q.T @ W @ Q
    # 添加正则项确保正定性
    C += np.eye(4) * eps
    vals, vecs = eigh(C)
    return quat_normalize(vecs[:, -1])

def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

def main():
    xs = np.random.randn(10,4)
    w = np.random.random_integers(1,100,10)
    w = w/np.linalg.norm(w)

    logger.debug(f"{weightedAverageQuaternions(xs,w)=}")
    logger.debug(f"{weighted_quat_mean(xs,w)=}")
    import torch
    torch.distributions.MixtureSameFamily

if __name__=='__main__':
    main()