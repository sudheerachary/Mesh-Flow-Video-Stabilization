import numpy as np
from cvxpy import *
from tqdm import tqdm
from multiprocessing import Pool

def gauss(t, r, window_size):
    """
    @param: window_size is the size of window over which gaussian to be applied
    @param: t is the index of current point 
    @param: r is the index of point in window 
    
    Return:
            returns spacial guassian weights over a window size
    """
    if np.abs(r-t) > window_size:
        return 0
    else:
        return np.exp((-9*(r-t)**2)/window_size**2)


def offline_optimize_path(c, iterations=100, window_size=6):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term
    
    
    Returns:
            returns an optimized gaussian smooth camera trajectory 
    """
    lambda_t = 100
    p = np.empty_like(c)
    
    W = np.zeros((c.shape[2], c.shape[2]))
    for t in range(W.shape[0]):
        for r in range(-window_size/2, window_size/2+1):
            if t+r < 0 or t+r >= W.shape[1] or r == 0:
                continue
            W[t, t+r] = gauss(t, t+r, window_size)

    gamma = 1+lambda_t*np.dot(W, np.ones((c.shape[2],)))
    
    bar = tqdm(total=c.shape[0]*c.shape[1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            P = np.asarray(c[i, j, :])
            for iteration in range(iterations):
                P = np.divide(c[i, j, :]+lambda_t*np.dot(W, P), gamma)
            p[i, j, :] = np.asarray(P)
            bar.update(1)
    
    bar.close()
    return p

def real_time_optimize_path(c, buffer_size=200, iterations=10, window_size=32, beta=1):
    """
    @param: c is camera trajectory within the buffer

    Returns:
        returns an realtime optimized smooth camera trajectory
    """

    lambda_t = 100
    p = np.empty_like(c)
    
    W = np.zeros((buffer_size, buffer_size))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] = gauss(i, j, window_size)
    
    bar = tqdm(total=c.shape[0]*c.shape[1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            y = []; d = None
            # real-time optimization
            for t in range(1, c.shape[2]+1):
                if t < buffer_size+1:
                    P = np.asarray(c[i, j, :t])
                    if not d is None:
                        for _ in range(iterations):
                            alpha = c[i, j, :t] + lambda_t*np.dot(W[:t, :t], P)
                            alpha[:-1] = alpha[:-1] + beta*d
                            gamma = 1 + lambda_t*np.dot(W[:t, :t], np.ones((t,)))
                            gamma[:-1] = gamma[:-1] + beta
                            P = np.divide(alpha, gamma)
                else:
                    P = np.asarray(c[i, j, t-buffer_size:t])
                    for _ in range(iterations):
                        alpha = c[i, j, t-buffer_size:t] + lambda_t*np.dot(W, P)
                        alpha[:-1] = alpha[:-1] + beta*d[1:]
                        gamma = 1 + lambda_t*np.dot(W, np.ones((buffer_size,)))
                        gamma[:-1] = gamma[:-1] + beta
                        P = np.divide(alpha, gamma)
                d = np.asarray(P); y.append(P[-1])
            p[i, j, :] = np.asarray(y)
            bar.update(1)
    bar.close()
    return p

def parallel_optimize(vertex_profiles):
    """
    @param: vertex_profiles is the accumulation of the 
            motion vectors at the mesh vertices
    
    Return:
            returns a parallely optimized smooth vertex 
            profiles for all mesh vertices
    
    """
    pool = Pool(processes=20)
    args = list(product(range(vertex_profiles.shape[0]), range(vertex_profiles.shape[1])))
    paths = pool.map(optimize_path, [vertex_profiles[arg[0], arg[1]] for arg in args])
    return paths


def cvx_optimize_path(c, buffer_size=0, window_size=6):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term
    
    
    Returns:
            returns an optimized gaussian smooth camera trajectory 
    """
    lambda_t = 100
    if window_size > c.shape[2]:
        window_size = c.shape[2]
    
    p = np.empty_like(c)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            P = Variable(c.shape[2])
            for t in range(c.shape[2]):
                
                # first term for optimised path to be close to camera path
                path_term = (P[t]-c[i, j, t])**2

                # second term for smoothness using gaussian weights
                for r in range(window_size):
                    if t-r < 0:
                        break
                    w = gauss(t, t-r, window_size)
                    gauss_weight = w*(P[t]-P[t-r])**2
                    if r == 0:
                        gauss_term = gauss_weight
                    else:
                        gauss_term += gauss_weight

                if t == 0:
                    objective = path_term + lambda_t*gauss_term
                else:
                    objective += path_term + lambda_t*gauss_term
            prob = Problem(Minimize(objective))
            prob.solve()
            p[i, j, :] = np.asarray(P.value).reshape(-1)
    return p