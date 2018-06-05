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
    return np.exp((-9*(r-t)**2)/window_size**2)


def optimize_path(c, iterations=100, window_size=6):
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