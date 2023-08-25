import numpy as np
import math

def projectOnSimplex(v):
    """
    A fast and efficient projection algorithm to project onto the
    standard probability simplex.

    :param numpy.ndarray v: Vector to project onto the simplex.
    :returns: Projection of the input onto the standard probability simplex.
    :rtype numpy.ndarray
    """
    n, = v.shape
    # check if we are already on the simplex
    if v.sum() == 1 and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - 1) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def log_opt(v):
    """
    Solves the maximization problem sum_i log(x_i) + sum_i x_i * v_i
    subject to the constraint that x lies on the standard simplex

    :param numpy.ndarray v: Input vector to the above optimization problem.
    :returns: A tuple containing point on the simplex attaining the optimum and the optimal value.
    :rtype tuple
    """

    est = -v.max() - 1
    f_val = 1

    while(abs(f_val) >= 1e-9 or max(v + est) > 0):
        f_val = (1 / (v + est)).sum() + 1
        f_dash = -(((1 / (v + est))**2).sum())
        est = est - f_val / f_dash

    opt_point = -(1 / (v + est))
    opt_point = projectOnSimplex(opt_point)     # for safety, project onto simplex
    opt_val = np.log(opt_point).sum() + np.inner(opt_point, v)
    return (opt_point, opt_val)

def jains_fairness_index(v):
    """
    Return Jain's Fairness Index, computed on the coordinates of the vector ``v``.

    :param numpy.ndarray v: Vector to compute the index on.
    :returns: Jain's fairness index.
    :rtype: float
    """
    n = v.shape[0]
    return ((v.sum())**2) / (n * (v * v).sum())

def ftrlOptimize(v, eta):
    """
    Performs the optimization step for the Follow the Regularized Leader
    (FTRL) framework with entropic regularizer over the standard probability simplex. Returns
    a point ``p`` which maximizes ``p * v - p * log(p)`` along with the optimal value. See Algorithm 6
    of the paper "k-experts - Online Policies and Fundamental Limits" for the algorithm.

    :param cumulativeGradient: Total gradient seen until the previous iteration.
    :param float eta: Learning rate.
    :returns: Tuple containing point ``p`` in the standard simplex attaining the optimum and the optimal value.
    :rtype: tuple

    """
    N = v.shape[0]

    # sort cumulativeGradient in non-increasing order
    orderedVector = -np.sort(-v)

    # finding i_star
    i_star = N
    tail_sum = 0
    while i_star >= 1:
        if (1 - i_star)*math.exp(eta*orderedVector[i_star - 1]) >= tail_sum:
            break
        else:
            tail_sum = tail_sum + math.exp(eta*orderedVector[i_star - 1])
            i_star = i_star - 1

    # computing K
    if i_star == N:     # we will have k = N in this case
        return np.ones(shape=N)

    # assuming that i_star < N
    K = (1 - i_star)/tail_sum
    p = np.zeros(shape=N)
    for i in range(1, N + 1):
        p[i - 1] = min(1, K*math.exp(eta*orderedVector[i - 1]))
    return p
